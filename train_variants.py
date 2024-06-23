from typing import Callable
from functools import partial
import os

import time
import argparse
import jax
import jax.numpy as jnp

from torch.utils.data import Dataset, Sampler, DataLoader
import torch

import numpy as np

import flax
import flax.linen as nn

import optax
import pickle

import tqdm

from functions import Fourier, Mixture, Slope, Polynomial, WhiteNoise, Shift
from networks import MixtureNeuralProcess, MLP, MeanAggregator, SequenceAggregator, NonLinearMVN, ResBlock


class ProbabilitySampler(Sampler):
    def __init__(self, probs):
        self.probs = probs / np.sum(probs)  # Ensure probabilities sum to 1

    def __iter__(self):
        # Sample indices according to the probabilities
        p = self.probs
        p = np.asarray(p).astype('float64')
        if p.sum() != 0:
            p = p * (1. / p.sum())
        indices = np.random.choice(len(self.probs), size=len(self.probs), replace=True, p=p)
        return iter(indices)

    def __len__(self):
        return len(self.probs)

    def update_probs(self, new_probs):
        self.probs = new_probs / np.sum(new_probs)  # Update and normalize probabilities


class SimpleDataset(Dataset):
    def __init__(self, dataset, context_size, dataset_size):
        self.context_size = context_size
        self.dataset_size = dataset_size
        self.context_xs, self.target_xs, self.context_ys, self.target_ys, self.distribs, self.noises = self._get_data(
            dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.context_xs[idx], self.context_ys[idx], self.target_xs[idx], self.target_ys[idx], self.distribs[idx], \
        self.noises[idx], idx

    def _get_data(self, dataset):
        xs_ys, distribs, noises = dataset
        xs, ys = xs_ys
        context_xs, target_xs = jnp.split(xs, indices_or_sections=(self.context_size,), axis=1)
        context_ys, target_ys = jnp.split(ys, indices_or_sections=(self.context_size,), axis=1)
        return context_xs, target_xs, context_ys, target_ys, distribs, noises


def f(
        key: flax.typing.PRNGKey,
        x: jax.Array,
        noise_scale: float = 0.2,
        mixture_prob: float = 0.5,
        corrupt: bool = True
):
    key_noise, key_mixture = jax.random.split(key)

    noise = jax.random.normal(key, x.shape) * noise_scale
    choice = jax.random.bernoulli(key_mixture, mixture_prob, x.shape)

    # return choice * (jnp.sin(2 * jnp.pi * x / 2)) + (1 - choice) * (jnp.cos(2 * jnp.pi * 2 * x)) + corrupt * noise
    return choice * (-2 - jnp.cos(2 * jnp.pi * x)) + (1 - choice) * (2 + jnp.cos(2 * jnp.pi * x)) + corrupt * noise


@jax.jit
def get_prob_score(variance, count):
    return jax.lax.cond(count >= 2, lambda _: (jnp.sqrt(variance + (variance ** 2 / (count - 1)))), lambda _: 0.0, 0)


@jax.jit
def get_prob_scores(variances, counts, eps=0.05):
    probs = jax.vmap(get_prob_score)(variances, counts)
    return probs + eps


def numpy_collate(batch):
    transposed_data = list(zip(*batch))
    xs_context = np.array(transposed_data[0])
    ys_context = np.array(transposed_data[1])
    xs_target = np.array(transposed_data[2])
    ys_target = np.array(transposed_data[3])
    distrib = np.array(transposed_data[4])
    noise = np.array(transposed_data[5])
    idx = np.array(transposed_data[6], dtype=np.int32)
    return torch.tensor(xs_context), torch.tensor(ys_context), torch.tensor(xs_target), torch.tensor(
        ys_target), torch.tensor(distrib), torch.tensor(noise), torch.tensor(idx)


@jax.jit
def add_err(x, count, m, s):
    x = jnp.squeeze(x)
    new_count = count + 1
    new_m = jax.lax.cond((count > 1), lambda _: (m + (x - m) / count), lambda _: x, 0)
    new_s = jax.lax.cond((count > 1), lambda _: (s + (x - m) * (x - new_m)), lambda _: 0.0, 0)
    variance = jax.lax.cond((count > 1), lambda _: (new_s / (count - 1)), lambda _: 0.0, 0)
    return variance, new_count, new_m, new_s


# @partial(jax.jit, static_argnums=(5,))
# def update_vars(counts, ms, ss, vars, new_val, ind):
#     count = counts[ind]
#     m = ms[ind]
#     s = ss[ind]
#     variance, new_count, new_m, new_s = add_err(new_val, count, m, s)
#     new_ms = ms.at[ind].set(new_m)
#     new_ss = ss.at[ind].set(new_s)
#     new_counts = counts.at[ind].set(new_count)
#     new_vars = vars.at[ind].set(variance)
#     return new_vars, new_ms, new_ss, new_counts
#

def initialize_model():
    embedding_xs = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_ys = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)
    embedding_both = MLP([64, 64], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True)

    projection_posterior = NonLinearMVN(
        MLP([128, 64], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True))

    output_model = nn.Sequential([
        ResBlock(
            MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
        ),
        ResBlock(
            MLP([128, 128], activation=jax.nn.leaky_relu, activate_final=True, use_layernorm=True),
        ),
        nn.Dense(2)
    ])
    projection_outputs = NonLinearMVN(output_model)

    posterior_aggregator = MeanAggregator(projection_posterior)

    model = MixtureNeuralProcess(
        embedding_xs, embedding_ys, embedding_both,
        posterior_aggregator,
        projection_outputs
    )
    return model


def initialize_params(model, rng, dataset_size):
    rng, key_data, key_test, key_x = jax.random.split(rng, 4)

    keys_data = jax.random.split(key_data, (dataset_size,))

    xs = jax.random.uniform(key_x, (dataset_size,)) * 2 - 1
    ys = jax.vmap(f)(keys_data, xs)
    rng, key1, key2 = jax.random.split(rng, 3)
    params = model.init({'params': key1, 'default': key2}, xs[:, None], ys[:, None], xs[:3, None])
    return params


@jax.jit
def batch_to_screenernet_input(xs, ys):
    xs = xs[:, :, 0]
    ys = ys[:, :, 0]
    return jnp.concatenate((xs, ys), axis=1)


@partial(jax.jit, static_argnums=(2,))
def screenernet_loss(screenernet, screenernet_input, apply_fn, losses):
    """
    Computes the objective loss of ScreenerNet.
    """
    weights = apply_fn(screenernet, screenernet_input).flatten()

    def body_fun(i, loss_sn):
        loss = losses[i]
        weight = weights[i]  # what is the value?
        regularization_term = (1 - weight) * (1 - weight) * loss + weight * weight * jnp.maximum(1.5 - loss, 0)
        return loss_sn + regularization_term

    # flat_loss = jnp.sum(jnp.abs(flattened))
    loss_screenernet = 0.0
    loss_screenernet = jax.lax.fori_loop(0, len(losses), body_fun, loss_screenernet)
    # loss_screenernet = loss_screenernet * (1 / len(losses)) + alpha * flat_loss
    loss_screenernet = loss_screenernet * (1 / len(losses))
    return loss_screenernet


@partial(jax.jit, static_argnums=(0, 1, 2, 9, 10))
def np_losses_batch_elbo(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context, xs_target, ys_target,
                         key, kl_penalty, num_posterior_mc):
    """
    Computes the un-weighted ELBOs for all tasks in a batch.
    """
    # Compute ELBO over batch of datasets
    elbos = jax.vmap(partial(
        apply_fn,
        np_params,
        beta=kl_penalty, k=num_posterior_mc,
        method=elbo_fn
    ))(
        xs_context, ys_context, xs_target, ys_target, rngs={'default': jax.random.split(key, f_size)}
    )
    return elbos


@jax.jit
def elementwise_gaussian_ll_loss(y, mean, std):
    eps = 1e-6
    v = std * std
    return jnp.log(jnp.maximum(v, eps)) + (y - mean) ** 2 / jnp.maximum(eps, v)


@jax.jit
def sample_gaussian_ll_loss(ys, means, stds):
    losses = jax.vmap(elementwise_gaussian_ll_loss, in_axes=(0, 0, 0))(ys, means, stds)
    res = 0.5 * jnp.mean(losses)
    return res


@partial(jax.jit, static_argnums=(0, 1, 2, 10, 11))
def np_weighted_loss_elbo(apply_fn, elbo_fn, f_size, np_params, weights, xs_context, ys_context, xs_target,
                          ys_target, key, kl_penalty, num_posterior_mc):
    """
    Computes the weighted loss for a batch of tasks.
    """
    elbos = np_losses_batch_elbo(apply_fn, elbo_fn, f_size, np_params, xs_context, ys_context,
                                 xs_target, ys_target, key, kl_penalty, num_posterior_mc)
    weighted_elbos = elbos * weights
    return -weighted_elbos.mean()  # try just *


@partial(jax.jit, static_argnums=(0, 1, 2, 11, 12, 13))
def update_np_elbo(
        apply_fn,
        elbo_fn,
        f_size,
        theta: flax.typing.VariableDict,
        opt_state: optax.OptState,
        weights,
        xs_context,
        ys_context,
        xs_target,
        ys_target,
        random_key: flax.typing.PRNGKey,
        optimizer,
        kl_penalty,
        num_posterior_mc
) -> tuple[flax.typing.VariableDict, optax.OptState, jax.Array]:
    # Implements a generic SGD Step

    value, grad = (jax.value_and_grad(np_weighted_loss_elbo, argnums=3)
                   (apply_fn, elbo_fn, f_size, theta, weights, xs_context, ys_context, xs_target, ys_target,
                    random_key, kl_penalty, num_posterior_mc))

    updates, opt_state = optimizer.update(grad, opt_state, theta)
    theta = optax.apply_updates(theta, updates)

    return theta, opt_state, value


@partial(jax.jit, static_argnums=(0, 3))
def update_screenernet(tx, screenernet_opt, screenernet_input, apply_fn, screenernet, losses):
    """
    Performs one gradient step on the ScreenerNet.
    """
    loss_grad_fn = jax.value_and_grad(screenernet_loss, argnums=0)
    loss_val, grads = loss_grad_fn(screenernet, screenernet_input, apply_fn, losses)
    updates, opt_state = tx.update(grads, screenernet_opt)
    screenernet = optax.apply_updates(screenernet, updates)
    return loss_val, screenernet


@partial(jax.jit, static_argnums=(0,))
def evaluate(apply_fn, np_params, key, batch):
    X, y, x_test, y_test, distrib, noise, idx = batch
    X = X.reshape((X.shape[1], X.shape[0], X.shape[2]))
    y = y.reshape((y.shape[1], y.shape[0], y.shape[2]))
    x_test = x_test.reshape((x_test.shape[1], x_test.shape[0], x_test.shape[2]))
    y_test = y_test.reshape((y_test.shape[1], y_test.shape[0], y_test.shape[2]))
    # key_ll, key_eval = jax.random.split(key)
    print(X.shape, y.shape, x_test.shape, y_test.shape)
    means, stds = apply_fn(
        np_params,
        X[:, None], y[:, None], x_test[:, None],
        k=1,
        rngs={'default': key}
    )
    # keys = jax.random.split(key_ll, y_test.shape[0])
    L = sample_gaussian_ll_loss(y_test, means, stds)
    return L

# def add_err(x, count, m, s):
#     new_count = count + 1
#     new_m = jax.lax.cond((count > 1), lambda _: (m + (x - m) / count), lambda _: x, 0)
#     new_s = jax.lax.cond((count > 1), lambda _: (s + (x - m) * (x - new_m)), lambda _: 0.0, 0)
#     variance = jax.lax.cond((count > 1), lambda _: (new_s / (count - 1)), lambda _: 0.0, 0)
#     return variance, new_count, new_m, new_s

@jax.jit
def batch_update_errs(losses, indices, ms, ss, counts, all_vars):
    indices = jnp.array(indices)
    losses_arr = jnp.array(losses.reshape((128, 1)))
    batched_f = jax.vmap(add_err, in_axes=(0, 0, 0, 0))
    batch_counts = counts[indices]
    batch_ms = ms[indices]
    batch_ss = ss[indices]
    batch_vars, new_batch_counts, new_batch_ms, new_batch_ss = batched_f(losses_arr, batch_counts, batch_ms, batch_ss)
    new_counts = counts.at[indices].set(new_batch_counts)
    new_vars = all_vars.at[indices].set(batch_vars)
    new_ms = ms.at[indices].set(new_batch_ms)
    new_ss = ss.at[indices].set(new_batch_ss)
    return new_vars, new_ms, new_ss, new_counts



def generate_datasets_with_setup(dataset_size, setup, target_size, context_size, key):
    key_test, key_train = jax.random.split(key)
    noise_levels1 = [0.0, 0.1, 0.2]
    noise_levels2 = [0.01, 0.01, 0.01]
    sampler_ratio = [0.5, 0.25, 0.25]
    fourier = Fourier(n=2, amplitude=.5, period=1.0)
    slope = Slope()
    polynomial2 = Polynomial(order=2, clip_bounds=(-1, 1))
    FOURIER = 0
    POLYN = 1
    SLOPE = 2

    data_sampler1_1 = partial(joint, WhiteNoise(fourier, noise_levels1[0]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler1_2 = partial(joint, WhiteNoise(fourier, noise_levels1[1]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler1_3 = partial(joint, WhiteNoise(fourier, noise_levels1[2]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler_props1_1 = {
        "distribution": FOURIER,
        "noise": noise_levels1[0],
        "sampler": data_sampler1_1
    }
    data_sampler_props1_2 = {
        "distribution": FOURIER,
        "noise": noise_levels1[1],
        "sampler": data_sampler1_2
    }
    data_sampler_props1_3 = {
        "distribution": FOURIER,
        "noise": noise_levels1[2],
        "sampler": data_sampler1_3
    }

    data_sampler2_1 = partial(joint, WhiteNoise(fourier, noise_levels2[0]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler2_2 = partial(joint, WhiteNoise(slope, noise_levels2[1]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler2_3 = partial(joint, WhiteNoise(polynomial2, noise_levels2[2]),
                              partial(uniform, n=context_size + target_size, bounds=(-1, 1)))
    data_sampler_props2_1 = {
        "distribution": FOURIER,
        "noise": noise_levels2[0],
        "sampler": data_sampler2_1
    }
    data_sampler_props2_2 = {
        "distribution": SLOPE,
        "noise": noise_levels2[1],
        "sampler": data_sampler2_2
    }
    data_sampler_props2_3 = {
        "distribution": POLYN,
        "noise": noise_levels2[2],
        "sampler": data_sampler2_3
    }
    if setup == 1:
        dataset_train = SimpleDataset(
            generate_noisy_split_trainingdata([data_sampler_props1_1, data_sampler_props1_2, data_sampler_props1_3],
                                              sampler_ratio, dataset_size, key_train), context_size=context_size,
            dataset_size=dataset_size)
        dataset_test = SimpleDataset(
            generate_noisy_split_trainingdata([data_sampler_props1_1, data_sampler_props1_2, data_sampler_props1_3],
                                              sampler_ratio, dataset_size, key_test), context_size=context_size,
            dataset_size=dataset_size)
        return dataset_train, dataset_test
    else:
        dataset_train = SimpleDataset(
            generate_noisy_split_trainingdata([data_sampler_props2_1, data_sampler_props2_2, data_sampler_props2_3],
                                              sampler_ratio, dataset_size, key_train), context_size=context_size,
            dataset_size=dataset_size)
        dataset_test = SimpleDataset(
            generate_noisy_split_trainingdata([data_sampler_props2_1, data_sampler_props2_2, data_sampler_props2_3],
                                              sampler_ratio, int(0.1 * dataset_size), key_test), context_size=context_size,
            dataset_size=int(0.1 * dataset_size))
        return dataset_train, dataset_test


def save(path, version, time_str, method, losses, setup, params):
    os.makedirs(os.path.join(path, time_str), exist_ok=True)
    pickle_dict = {
        'losses': losses,
        'params': params
    }
    name = 'evolution_results_' + method + '_' + str(setup) + '_' + str(version)
    with open(os.path.join(path, time_str, name + '.pkl'), 'wb') as f:
        pickle.dump(pickle_dict, f)


def save_evolution(path, version, time_str, method, setup, values):
    os.makedirs(os.path.join(path, time_str), exist_ok=True)
    name = 'ev_' + method + '_' + str(setup) + '_' + str(version)
    with open(os.path.join(path, time_str, name + '.pkl'), 'wb') as f:
        pickle.dump(values, f)
def train_baseline(random_seed, dir_path, setup):
    # declare constants
    batch_size = 128
    context_size = 64
    target_size = 32
    num_epochs = 150
    kl_penalty = 1e-4
    num_posterior_mc = 1
    rng = jax.random.key(random_seed)
    test_resolution = 512
    dataset_size = 128 * 1
    time_str = str(time.time())
    # declare models
    model = initialize_model()
    optimizer = optax.chain(
        optax.clip(.1),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
    )
    for version in range(5):
        # declare loss arrays
        baseline_losses = list()
        # run 5 times training for NP, NP+SN, NP+AB, NP+SN+AB
        # declare dataset set-up

        key_baseline, rng = jax.random.split(rng)
        baseline_params = initialize_params(model, key_baseline, dataset_size)
        opt_state_baseline = optimizer.init(baseline_params)
        # best params
        best_np_loss, best_np_params = jnp.inf, baseline_params

        key, rng = jax.random.split(rng)
        dataset_train, dataset_test = generate_datasets_with_setup(dataset_size, setup, target_size, context_size, key)
        for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
            dl = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, collate_fn=numpy_collate)
            test_dl = DataLoader(dataset_test, shuffle=True, batch_size=1, collate_fn=numpy_collate)
            data_it = iter(dl)
            for stp in range(int(dataset_size / batch_size)):
                batch = next(data_it)
                batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                xs_context, ys_context, xs_target, ys_target, distrib, noise, idx = batch
                # update baseline
                key, rng = jax.random.split(rng)
                baseline_params, opt_state_baseline, loss_np = update_np_elbo(model.apply,
                                                                                    model.elbo, batch_size,
                                                                                    baseline_params, opt_state_baseline,
                                                                                    jnp.ones(batch_size), xs_context,
                                                                                    ys_context, xs_target, ys_target,
                                                                                    key, optimizer, kl_penalty,
                                                                                    num_posterior_mc)
                # store best params
                if loss_np < best_np_loss:
                    best_np_loss = loss_np
                    best_np_params = baseline_params
            if epoch % 5 == 4:
                avg_loss_np = 0.0
                cntr = 0
                for batch in iter(test_dl):
                    cntr += 1
                    key, rng = jax.random.split(rng)
                    batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                    avg_loss_np = avg_loss_np + evaluate(model.apply, best_np_params, key, batch)
                baseline_losses.append(avg_loss_np / cntr)
                save(dir_path, version, time_str, method='baseline', losses=baseline_losses, params=best_np_params, setup=setup)


def train_sn(random_seed, dir_path, setup):
    # declare constants
    batch_size = 128
    context_size = 64
    target_size = 32
    num_epochs = 150
    kl_penalty = 1e-4
    num_posterior_mc = 1
    rng = jax.random.key(random_seed)
    test_resolution = 512
    dataset_size = 128 * 100
    time_str = str(time.time())
    model = initialize_model()
    # declare weight arrays
    tracked_indices = list()
    all_weights_sn = jax.numpy.zeros((int(0.9 * num_epochs / 5), dataset_size))
    sn_model = nn.Sequential([
        MLP([2 * context_size, 64, 64, 128], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True),
        MLP([128, 1], activation=jax.nn.relu, activate_final=True, use_layernorm=False)
    ])
    dummy = jnp.zeros(2 * context_size, )
    tx = optax.adam(learning_rate=1e-3)
    for version in range(1):
        key, rng = jax.random.split(rng)
        screenernet_params = sn_model.init(key, dummy)
        sn_opt_state = tx.init(screenernet_params)
        key, rng = jax.random.split(rng)
        npsn_params = initialize_params(model, key, dataset_size)
        optimizer = optax.chain(
            optax.clip(.1),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
        )
        opt_state_npsn = optimizer.init(npsn_params)
        # declare loss arrays
        npsn_losses = list()
        # best params
        best_sn_loss, best_sn_params = jnp.inf, npsn_params
        # run 5 times training for NP, NP+SN, NP+AB, NP+SN+AB
        # declare dataset set-up
        key, rng = jax.random.split(rng)
        dataset_train, dataset_test = generate_datasets_with_setup(dataset_size, setup, target_size, context_size, key)
        for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
            dl = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, collate_fn=numpy_collate)
            test_dl = DataLoader(dataset_test, shuffle=True, batch_size=1, collate_fn=numpy_collate)
            data_it = iter(dl)
            for stp in range(int(dataset_size / batch_size)):
                batch = next(data_it)
                batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                xs_context, ys_context, xs_target, ys_target, distrib, noise, idx = batch
                screenernet_input = batch_to_screenernet_input(xs_context, ys_context)
                # get losses
                key, rng = jax.random.split(rng)
                losses_sn = np_losses_batch_elbo(model.apply, model.err, batch_size, npsn_params, xs_context,
                                                 ys_context, xs_target, ys_target, key, kl_penalty, num_posterior_mc)
                # get weights and normalize
                weights_sn = sn_model.apply(screenernet_params, screenernet_input).flatten()
                if epoch < num_epochs / 10:
                    weights_sn = jnp.ones(weights_sn.shape)
                else:
                    sum_weights_sn = jnp.sum(weights_sn, axis=None)
                    if sum_weights_sn != 0:
                        weights_sn = (batch_size / sum_weights_sn) * weights_sn
                    if epoch % 5 == 4:
                        update_ind = int((epoch - 4 - (num_epochs / 10)) / 5)
                        all_weights_sn = all_weights_sn.at[update_ind, idx].set(weights_sn)
                # update sn
                key, rng = jax.random.split(rng)
                npsn_params, opt_state_npsn, loss_npsn = update_np_elbo(model.apply, model.elbo, batch_size,
                                                                               npsn_params, opt_state_npsn, weights_sn,
                                                                               xs_context,
                                                                               ys_context, xs_target, ys_target, key,
                                                                               optimizer, kl_penalty, num_posterior_mc)
                loss_sn, screenernet_params = update_screenernet(tx, sn_opt_state, screenernet_input,
                                                                 sn_model.apply, screenernet_params, losses_sn)
                # store best params
                if loss_npsn < best_sn_loss:
                    best_sn_params = npsn_params
                    best_sn_loss = loss_npsn
            if epoch % 5 == 4:
                if epoch in [29, 99, 149]:
                    tracked_indices.append(jnp.argmax(all_weights_sn[int((epoch - 4 - (num_epochs / 10)) / 5)]))
                avg_loss_npsn = 0.0
                cntr = 0
                for batch in iter(test_dl):
                    cntr += 1
                    key, rng = jax.random.split(rng)
                    batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                    avg_loss_npsn = avg_loss_npsn + evaluate(model.apply, best_sn_params, key, batch)
                npsn_losses.append(avg_loss_npsn / cntr)
                save(dir_path, version, time_str, method='screenernet', losses=npsn_losses, setup=setup, params=best_sn_params)
    save_evolution(dir_path, version, time_str, method='sn_weights', setup=setup, values=all_weights_sn)

def train_ab(random_seed, dir_path, setup):
    # declare constants
    batch_size = 128
    context_size = 64
    target_size = 32
    num_epochs = 150
    kl_penalty = 1e-4
    num_posterior_mc = 1
    rng = jax.random.key(random_seed)
    test_resolution = 512
    dataset_size = 128 * 100
    time_str = str(time.time())
    model = initialize_model()
    all_scores_ab = jax.numpy.zeros((int(0.5 * num_epochs / 5), dataset_size))
    tracked_indices = list()
    # run 5 times training for NP, NP+SN, NP+AB, NP+SN+AB
    for version in range(1):
        # declare loss arrays
        npab_losses = list()
        # declare ab utils
        ms_ab = jnp.zeros(dataset_size)
        ss_ab = jnp.zeros(dataset_size)
        counts_ab = jnp.zeros(dataset_size)
        all_vars_ab = jnp.zeros(dataset_size)
        batch_sampler_ab = ProbabilitySampler(jnp.ones(dataset_size))
        # declare dataset set-up
        key, rng = jax.random.split(rng)
        dataset_train, dataset_test = generate_datasets_with_setup(dataset_size, setup, target_size, context_size, key)
        key, rng = jax.random.split(rng)
        npab_params = initialize_params(model, key, dataset_size)
        # best params
        best_ab_loss, best_ab_params = jnp.inf, npab_params
        optimizer = optax.chain(
            optax.clip(.1),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
        )
        opt_state_npab = optimizer.init(npab_params)
        for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
            test_dl = DataLoader(dataset_test, shuffle=True, batch_size=1, collate_fn=numpy_collate)
            if epoch >= int(num_epochs * 0.50):
                scores_ab = get_prob_scores(all_vars_ab, counts_ab)
                if epoch % 5 == 4:
                    update_ind = int((epoch - 4 - (num_epochs / 2)) / 5)
                    all_scores_ab = all_scores_ab.at[update_ind].set(scores_ab)
                    if epoch in [79, 99, 119, 149]:
                        tracked_indices.append(jnp.argmax(all_scores_ab[update_ind]))
                batch_sampler_ab.update_probs(scores_ab)
            dl_ab = DataLoader(dataset_train, sampler=batch_sampler_ab, batch_size=batch_size, collate_fn=numpy_collate)
            data_it_ab = iter(dl_ab)
            for stp in range(int(dataset_size / batch_size)):
                batch_ab = next(data_it_ab)
                batch_ab = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch_ab)
                xs_context_ab, ys_context_ab, xs_target_ab, ys_target_ab, distrib_ab, noise_ab, idx_ab = batch_ab
                # get losses
                key, rng = jax.random.split(rng)
                losses_ab = np_losses_batch_elbo(model.apply, model.elbo, batch_size, npab_params, xs_context_ab,
                                                 ys_context_ab, xs_target_ab, ys_target_ab, key, kl_penalty,
                                                 num_posterior_mc)
                # get variances
                all_vars_ab, ms_ab, ss_ab, counts_ab = batch_update_errs(losses_ab, idx_ab, ms_ab, ss_ab, counts_ab,
                                                                         all_vars_ab)
                # update ab
                key, rng = jax.random.split(rng)
                npab_params, opt_state_npab, loss_npab = update_np_elbo(model.apply, model.elbo, batch_size, npab_params,
                                                               opt_state_npab, jnp.ones(batch_size), xs_context_ab, ys_context_ab, xs_target_ab,
                                                               ys_target_ab, key, optimizer, kl_penalty, num_posterior_mc)
                # store best params
                if loss_npab < best_ab_loss:
                    best_ab_params = npab_params
                    best_ab_loss = loss_npab
            if epoch % 5 == 4:
                avg_loss_ab = 0.0
                cntr = 0
                for batch in iter(test_dl):
                    cntr += 1
                    key, rng = jax.random.split(rng)
                    batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                    avg_loss_ab = avg_loss_ab + evaluate(model.apply, best_ab_params, key, batch)
                npab_losses.append(avg_loss_ab / cntr)
                save(dir_path, version, time_str, method='active_bias', losses=npab_losses, params=best_ab_params, setup=setup)
                save_evolution(dir_path, version, time_str, 'Active Bias', setup, all_scores_ab)



def train_snab(random_seed, dir_path, setup):
    # declare constants
    batch_size = 128
    context_size = 64
    target_size = 32
    num_epochs = 150
    kl_penalty = 1e-4
    num_posterior_mc = 1
    rng = jax.random.key(random_seed)
    test_resolution = 512
    dataset_size = 128 * 100
    time_str = str(time.time())
    # declare models
    model = initialize_model()
    optimizer = optax.chain(
        optax.clip(.1),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-6),
    )
    sn_model = nn.Sequential([
        MLP([2 * context_size, 64, 64, 128], activation=jax.nn.leaky_relu, activate_final=False, use_layernorm=True),
        MLP([128, 1], activation=jax.nn.relu, activate_final=True, use_layernorm=False)
    ])
    dummy = jnp.zeros(2 * context_size, )
    tx = optax.adam(learning_rate=1e-3)
    # run 5 times training for NP, NP+SN, NP+AB, NP+SN+AB
    for version in range(1):
        # declare loss arrays
        npsnab_losses = list()
        # declare weight arrays
        all_weights_snab = list()
        all_scores_snab = list()
        # declare ab utils
        ms_snab = jnp.zeros(dataset_size)
        ss_snab = jnp.zeros(dataset_size)
        counts_snab = jnp.zeros(dataset_size)
        all_vars_snab = jnp.zeros(dataset_size)
        batch_sampler_snab = ProbabilitySampler(jnp.ones(dataset_size))
        key, rng = jax.random.split(rng)
        npsnab_params = initialize_params(model, key, dataset_size)
        key, rng = jax.random.split(rng)
        screenernet_ab_params = sn_model.init(key, dummy)
        opt_state_npsnab = optimizer.init(npsnab_params)
        snab_opt_state = tx.init(screenernet_ab_params)
        # best params
        best_snab_loss, best_snab_params = jnp.inf, npsnab_params
        # declare dataset set-up
        key, rng = jax.random.split(rng)
        dataset_train, dataset_test = generate_datasets_with_setup(dataset_size, setup, target_size, context_size, key)
        for epoch in (pbar := tqdm.trange(num_epochs, desc='Optimizing params. ')):
            test_dl = DataLoader(dataset_test, shuffle=True, batch_size=1, collate_fn=numpy_collate)
            if epoch >= int(num_epochs * 0.5):
                scores_snab = get_prob_scores(all_vars_snab, counts_snab)
                all_scores_snab = scores_snab
                batch_sampler_snab.update_probs(scores_snab)
            dl_snab = DataLoader(dataset_train, sampler=batch_sampler_snab, batch_size=batch_size,
                                 collate_fn=numpy_collate)
            data_it_snab = iter(dl_snab)
            for stp in range(int(dataset_size / batch_size)):
                batch_snab = next(data_it_snab)
                batch_snab = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch_snab)
                xs_context_snab, ys_context_snab, xs_target_snab, ys_target_snab, distrib_snab, noise_snab, idx_snab = batch_snab
                screenernet_input_snab = batch_to_screenernet_input(xs_context_snab, ys_context_snab)
                # get losses
                key, rng = jax.random.split(rng)
                losses_snab = np_losses_batch_elbo(model.apply, model.err, batch_size, npsnab_params, xs_context_snab,
                                                   ys_context_snab, xs_target_snab, ys_target_snab, key, kl_penalty,
                                                   num_posterior_mc)
                # get variances
                all_vars_snab, ms_snab, ss_snab, counts_snab = batch_update_errs(losses_snab, idx_snab, ms_snab,
                                                                                 ss_snab, counts_snab, all_vars_snab)
                # get weights and normalize
                weights_snab = sn_model.apply(screenernet_ab_params, screenernet_input_snab).flatten()
                if epoch < num_epochs / 10:
                    weights_snab = jnp.ones(weights_snab.shape)
                else:
                    sum_weights_snab = jnp.sum(weights_snab, axis=None)
                    if sum_weights_snab != 0:
                        weights_snab = (batch_size / sum_weights_snab) * weights_snab
                    if stp % 10 == 9:
                        all_weights_snab.append(weights_snab)
                # update snab
                key, rng = jax.random.split(rng)
                npsnab_params, opt_state_npsnab, loss_npsnab = update_np_elbo(model.apply, model.elbo, batch_size, npsnab_params,
                                                               opt_state_npsnab, weights_snab, xs_context_snab, ys_context_snab, xs_target_snab,
                                                               ys_target_snab, key, optimizer, kl_penalty, num_posterior_mc)
                loss_snab, screenernet_ab_params = update_screenernet(tx, snab_opt_state, screenernet_input_snab,
                                                                 sn_model.apply, screenernet_ab_params, losses_snab)
                # store best params
                if loss_npsnab < best_snab_loss:
                    best_snab_params = npsnab_params
                    best_snab_loss = loss_npsnab
            if epoch % 5 == 4:
                avg_loss_snab = 0.0
                cntr = 0
                for batch in iter(test_dl):
                    cntr += 1
                    key, rng = jax.random.split(rng)
                    batch = jax.tree_util.tree_map(lambda tensor: tensor.numpy(), batch)
                    avg_loss_snab = avg_loss_snab + evaluate(model.apply, best_snab_params, key, batch)
                npsnab_losses.append(avg_loss_snab / cntr)
                save(dir_path, version, time_str, method='snab', losses=npsnab_losses, params=best_snab_params, setup=setup)
                save_evolution(dir_path, version, time_str, 'snab_scores', setup, all_scores_snab)
                save_evolution(dir_path, version, time_str, 'snab_weights', setup, all_weights_snab)

def generate_datasets(sampler_ratio_train, sampler_ratio_test, dataset_size, batch_size, target_size, context_size, key,
                      noise_levels):
    key_test, key_train = jax.random.split(key)
    f2 = Fourier(n=2, amplitude=.5, period=1.0)
    f5 = Slope()
    f6 = Polynomial(order=2, clip_bounds=(-1, 1))
    FOURIER = 0
    POLYN = 1
    SLOPE = 2

    data_sampler1 = partial(
        joint,
        WhiteNoise(f2, noise_levels[0]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_1 = {
        "distribution": FOURIER,
        "noise": noise_levels[0],
        "sampler": data_sampler1
    }
    data_sampler2 = partial(
        joint,
        WhiteNoise(f5, noise_levels[1]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_2 = {
        "distribution": SLOPE,
        "noise": noise_levels[1],
        "sampler": data_sampler2
    }
    data_sampler3 = partial(
        joint,
        WhiteNoise(f6, noise_levels[2]),
        partial(uniform, n=context_size + target_size, bounds=(-1, 1))
    )
    data_sampler_props_3 = {
        "distribution": POLYN,
        "noise": noise_levels[2],
        "sampler": data_sampler3
    }
    dataset_train = SimpleDataset(
        generate_noisy_split_trainingdata([data_sampler_props_1, data_sampler_props_2, data_sampler_props_3],
                                          sampler_ratio_train, dataset_size, key_train), context_size=context_size,
        dataset_size=dataset_size)
    dataset_test = SimpleDataset(
        generate_noisy_split_trainingdata([data_sampler_props_1, data_sampler_props_2, data_sampler_props_3],
                                          sampler_ratio_test, batch_size * 22, key_test), context_size=context_size,
        dataset_size=batch_size * 22)
    return dataset_train, dataset_test


def joint(
        module: nn.Module,
        data_sampler: Callable[
            [nn.Module, flax.typing.VariableDict, flax.typing.PRNGKey],
            tuple[jax.Array, jax.Array]
        ],
        key: flax.typing.PRNGKey,
        return_params: bool = False
) -> tuple[jax.Array, jax.Array]:
    # Samples from p(Z, X, Y)
    key_param, key_rng, key_data = jax.random.split(key, 3)

    params = module.init({'params': key_param, 'default': key_rng}, jnp.zeros(()))
    xs, ys = data_sampler(module, params, key_data)

    if return_params:
        return xs, ys, params
    return xs, ys


def uniform(
        module: nn.Module,
        params: flax.typing.VariableDict,
        key: flax.typing.PRNGKey,
        n: int,
        bounds: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    # Samples from p(X, Y | Z) = p(Y | Z, X)p(X)
    key_xs, key_ys = jax.random.split(key)
    xs = jax.random.uniform(key_xs, (n,)) * (bounds[1] - bounds[0]) + bounds[0]
    ys = jax.vmap(module.apply, in_axes=(None, 0))(params, xs, rngs={'default': jax.random.split(key_ys, n)})
    return xs, ys


@partial(jax.jit, static_argnums=(1))
def gen_sampler_datapoint(key, sampler):
    x, y = sampler(key)
    x, y = x[..., None], y[..., None]
    return x, y


@partial(jax.jit, static_argnums=(1, 2))
def generate_dataset(rng, num_batches, sampler):
    keys = jax.random.split(rng, num_batches)
    batched_generate = jax.vmap(partial(gen_sampler_datapoint, sampler=sampler))
    x, y = batched_generate(keys)
    return x, y


def generate_noisy_split_trainingdata(samplers, sampler_ratios, dataset_size, rng):
    """
    Generate a dataset with a split of different samplers and ratios
    """

    assert len(samplers) == len(sampler_ratios), "The number of samplers and ratios must be the same"
    assert sum(sampler_ratios) == 1.0, "The sum of the ratios must be 1.0"
    keys = jax.random.split(rng, len(samplers))
    datasets = []
    distribs = []
    noises = []
    for (sampler_prop, ratio, key) in zip(samplers, sampler_ratios, keys):
        sampler, distrib, noise = sampler_prop["sampler"], sampler_prop["distribution"], sampler_prop["noise"]
        dataset = generate_dataset(key, int(dataset_size * ratio), sampler)
        datasets.append(np.asarray(dataset))
        distribs.append(jnp.repeat(distrib, int(dataset_size * ratio)))
        noises.append(jnp.repeat(noise, int(dataset_size * ratio)))
    x_datasets, y_datasets = zip(*datasets)
    return np.asarray((jnp.concatenate(x_datasets), jnp.concatenate(y_datasets))), jnp.concatenate(
        distribs), jnp.concatenate(noises)


def main():
    parser = argparse.ArgumentParser(description="Parse command line arguments into specified values.")

    parser.add_argument("function_name", type=str, help="The name of the function to be called.")
    parser.add_argument("--seed", type=int, required=True, help="An integer value for the random seed.")
    parser.add_argument("--path", type=str, required=True, help="A string value for the path.")
    parser.add_argument("--setup", type=int, required=False, help="An integer value for the set-up type.", default=1)
    # parser.add_argument("--train_ratio", type=float, nargs=3, required=True,
    #                     help="An array of 3 decimal numbers for sampler ratio train.")
    # parser.add_argument("--test_ratio", type=float, nargs=3, required=True,
    #                     help="An array of 3 decimal numbers for sampler ratio test.")
    # parser.add_argument("--noise", type=float, nargs=3, required=True,
    #                     help="An array of 3 decimal numbers for noise levels.")

    args = parser.parse_args()

    # Extracting the arguments into the specified variables
    random_seed = args.seed
    path = args.path
    setup = args.setup
    # sampler_ratio_train = args.sampler_ratio_train
    # sampler_ratio_test = args.sampler_ratio_test
    # noise_levels = args.noise_levels

    # Assuming the function is defined in the global namespace and we want to call it
    if args.function_name in globals():
        globals()[args.function_name](random_seed, path, setup)
    else:
        print(f"Function {args.function_name} is not defined.")


if __name__ == "__main__":
    main()
