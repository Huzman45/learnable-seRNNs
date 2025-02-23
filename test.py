import neurogym as ngym
import gymnasium as gym
from neurogym import spaces
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.utils import scheduler
from neurogym.core import TrialWrapper

# Sanity check: See if original 20 Yang tasks are working with changes to the core neurogym functions
from neurogym.utils.scheduler import SequentialSchedule
import matplotlib.pyplot as plt
import numpy as np

from Mod_Cog.mod_cog_tasks import *

# tasks = ngym.get_collection("yang19")
# kwargs = {"dt": 100}
# envs = [gym.make(task, **kwargs) for task in tasks]
# schedule = SequentialSchedule(len(envs))
# env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
# dataset = ngym.Dataset(env, batch_size=1, seq_len=350)
# inputs, outputs = dataset()
# plt.imshow(inputs[:, 0, :].T)
# plt.plot(1 + outputs)
# plt.plot((outputs + 16) % 32 + 1)
# plt.xlim(0, 100)
# plt.show()
# print(tasks)

envs = [
    go(),
    rtgo(),
    dlygo(),
    anti(),
    rtanti(),
    dlyanti(),
    dm1(),
    dm2(),
    ctxdm1(),
    ctxdm2(),
    multidm(),
    dlydm1(),
    dlydm2(),
    ctxdlydm1(),
    ctxdlydm2(),
    multidlydm(),
    dms(),
    dnms(),
    dmc(),
    dnmc(),
    dlygointr(),
    dlygointl(),
    dlyantiintr(),
    dlyantiintl(),
    dlydm1intr(),
    dlydm1intl(),
    dlydm2intr(),
    dlydm2intl(),
    ctxdlydm1intr(),
    ctxdlydm1intl(),
    ctxdlydm2intr(),
    ctxdlydm2intl(),
    multidlydmintr(),
    multidlydmintl(),
    dmsintr(),
    dmsintl(),
    dnmsintr(),
    dnmsintl(),
    dmcintr(),
    dmcintl(),
    dnmcintr(),
    dnmcintl(),
    goseqr(),
    rtgoseqr(),
    dlygoseqr(),
    antiseqr(),
    rtantiseqr(),
    dlyantiseqr(),
    dm1seqr(),
    dm2seqr(),
    ctxdm1seqr(),
    ctxdm2seqr(),
    multidmseqr(),
    dlydm1seqr(),
    dlydm2seqr(),
    ctxdlydm1seqr(),
    ctxdlydm2seqr(),
    multidlydmseqr(),
    dmsseqr(),
    dnmsseqr(),
    dmcseqr(),
    dnmcseqr(),
    goseql(),
    rtgoseql(),
    dlygoseql(),
    antiseql(),
    rtantiseql(),
    dlyantiseql(),
    dm1seql(),
    dm2seql(),
    ctxdm1seql(),
    ctxdm2seql(),
    multidmseql(),
    dlydm1seql(),
    dlydm2seql(),
    ctxdlydm1seql(),
    ctxdlydm2seql(),
    multidlydmseql(),
    dmsseql(),
    dnmsseql(),
    dmcseql(),
    dnmcseql(),
]

env_analysis = MultiEnvs(envs, env_input=True)
schedule = RandomSchedule(len(envs))
env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
dataset = ngym.Dataset(env, batch_size=4, seq_len=350)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n


# To draw samples, use neurogym's dataset class:

inputs, labels = dataset()
