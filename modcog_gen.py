import neurogym as ngym
import gymnasium as gym
from neurogym import spaces
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.utils.scheduler import BaseSchedule

# Sanity check: See if original 20 Yang tasks are working with changes to the core neurogym functions
from neurogym.utils.scheduler import SequentialSchedule
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Mod_Cog.mod_cog_tasks import *


# The following are the original 20 tasks from Yang et al. (2019)
original_envs = {
    "go": go(),
    "rtgo": rtgo(),
    "dlygo": dlygo(),
    "anti": anti(),
    "rtanti": rtanti(),
    "dlyanti": dlyanti(),
    "dm1": dm1(),
    "dm2": dm2(),
    "ctxdm1": ctxdm1(),
    "ctxdm2": ctxdm2(),
    "multidm": multidm(),
    "dlydm1": dlydm1(),
    "dlydm2": dlydm2(),
    "ctxdlydm1": ctxdlydm1(),
    "ctxdlydm2": ctxdlydm2(),
    "multidlydm": multidlydm(),
    "dms": dms(),
    "dnms": dnms(),
    "dmc": dmc(),
    "dnmc": dnmc(),
}


intr_envs = {
    "dlygointr": dlygointr(),
    "dlyantiintr": dlyantiintr(),
    "dlydm1intr": dlydm1intr(),
    "dlydm2intr": dlydm2intr(),
    "ctxdlydm1intr": ctxdlydm1intr(),
    "ctxdlydm2intr": ctxdlydm2intr(),
    "multidlydmintr": multidlydmintr(),
    "dmsintr": dmsintr(),
    "dnmsintr": dnmsintr(),
    "dmcintr": dmcintr(),
    "dnmcintr": dnmcintr(),
}

intl_envs = {
    "dlygointl": dlygointl(),
    "dlyantiintl": dlyantiintl(),
    "dlydm1intl": dlydm1intl(),
    "dlydm2intl": dlydm2intl(),
    "ctxdlydm1intl": ctxdlydm1intl(),
    "ctxdlydm2intl": ctxdlydm2intl(),
    "multidlydmintl": multidlydmintl(),
    "dmsintl": dmsintl(),
    "dnmsintl": dnmsintl(),
    "dmcintl": dmcintl(),
    "dnmcintl": dnmcintl(),
}

int_envs = intr_envs | intl_envs

seqr_envs = {
    "goseqr": goseqr(),
    "rtgoseqr": rtgoseqr(),
    "dlygoseqr": dlygoseqr(),
    "antiseqr": antiseqr(),
    "rtantiseqr": rtantiseqr(),
    "dlyantiseqr": dlyantiseqr(),
    "dm1seqr": dm1seqr(),
    "dm2seqr": dm2seqr(),
    "ctxdm1seqr": ctxdm1seqr(),
    "ctxdm2seqr": ctxdm2seqr(),
    "multidmseqr": multidmseqr(),
    "dlydm1seqr": dlydm1seqr(),
    "dlydm2seqr": dlydm2seqr(),
    "ctxdlydm1seqr": ctxdlydm1seqr(),
    "ctxdlydm2seqr": ctxdlydm2seqr(),
    "multidlydmseqr": multidlydmseqr(),
    "dmsseqr": dmsseqr(),
    "dnmsseqr": dnmsseqr(),
    "dmcseqr": dmcseqr(),
    "dnmcseqr": dnmcseqr(),
}

seql_envs = {
    "goseql": goseql(),
    "rtgoseql": rtgoseql(),
    "dlygoseql": dlygoseql(),
    "antiseql": antiseql(),
    "rtantiseql": rtantiseql(),
    "dlyantiseql": dlyantiseql(),
    "dm1seql": dm1seql(),
    "dm2seql": dm2seql(),
    "ctxdm1seql": ctxdm1seql(),
    "ctxdm2seql": ctxdm2seql(),
    "multidmseql": multidmseql(),
    "dlydm1seql": dlydm1seql(),
    "dlydm2seql": dlydm2seql(),
    "ctxdlydm1seql": ctxdlydm1seql(),
    "ctxdlydm2seql": ctxdlydm2seql(),
    "multidlydmseql": multidlydmseql(),
    "dmsseql": dmsseql(),
    "dnmsseql": dnmsseql(),
    "dmcseql": dmcseql(),
    "dnmcseql": dnmcseql(),
}

seq_envs = seqr_envs | seql_envs

modcog_envs = original_envs | int_envs | seq_envs


def create_dataset(
    batch_size=128,
    seq_len=50,
    num_batches=80,
    envs=modcog_envs,
    train=True,
) -> ngym.Dataset:
    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=False)
    return ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len, batch_first=True)


class ModCogDataset:
    """
    A class to generate data for the ModCog tasks using neurogym.
    """

    def __init__(
        self,
        batch_size=4,
        seq_len=350,
        num_batches=80,
        envs=list(modcog_envs.values()),
        train=True,
    ):
        # env_analysis = MultiEnvs(envs, env_input=True)
        schedule = RandomSchedule(len(envs))
        self.env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        self.dataset = self._create_dataset(
            batch_size=batch_size,
            seq_len=seq_len,
            num_batches=num_batches,
            train=train,
        )

    def _batch_generator(self, dataset_instance, num_batches):
        """Generator to yield batches from a neurogym.Dataset instance."""
        for _ in range(num_batches):
            yield dataset_instance()  # dataset_instance() returns (inputs, labels)

    def _create_dataset(
        self, batch_size, seq_len, num_batches, train: bool
    ) -> tf.data.Dataset:
        task = ngym.Dataset(
            self.env, batch_size=batch_size, seq_len=seq_len, batch_first=True
        )

        example_batch = next(self._batch_generator(task, 1))
        example_inputs, example_labels = example_batch
        generator_output_shapes = (example_inputs.shape, example_labels.shape)
        generator_output_types = (example_inputs.dtype, example_labels.dtype)

        dataset = tf.data.Dataset.from_generator(
            lambda: self._batch_generator(
                task, num_batches
            ),  # Pass a callable that returns the generator
            output_types=generator_output_types,
            output_shapes=generator_output_shapes,
        )

        if train:
            dataset = dataset.shuffle(buffer_size=num_batches).repeat()

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class StaticSchedule(BaseSchedule):
    """Static schedules."""

    def __init__(self, n, i) -> None:
        super().__init__(n)
        self.fixed_i = i

    def __call__(self):
        return self.fixed_i


class SingleEnvDataset:
    """
    A class to generate data for the ModCog tasks using neurogym.
    """

    def __init__(
        self,
        batch_size=4,
        seq_len=350,
        full_envs=list(original_envs.values()),
        optimise_for_speed=True,
    ):
        self.optimise_for_speed = optimise_for_speed
        self.full_envs = full_envs
        self.num_envs = len(full_envs)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def generate_dataset(self, env_index: int = 0):
        schedule = StaticSchedule(self.num_envs, env_index)
        env = ScheduleEnvs(self.full_envs, schedule=schedule, env_input=True)
        raw_dataset = ngym.Dataset(
            env, batch_size=self.batch_size, seq_len=self.seq_len, batch_first=True
        )

        while True:
            yield raw_dataset()
