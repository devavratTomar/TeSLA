# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
import sys
from pathlib import Path
from utilities.utils import ensure_dir

import run_tta
import submitit


def parse_opt(module_):
    parser = argparse.ArgumentParser("Submitit for TeSLA", parents=[module_.get_opt_parser()])
    parser.add_argument("script", help="name of the script to run")
    parser.add_argument("--ngpus", default=2, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=15, type=int, help="Duration of the job")

    parser.add_argument("--partition", default="debug", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/home").is_dir():
        p = Path(f"/home/{user}/CVPR23/Experiments/init")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, opt, module_):
        self.opt = opt
        self.module_ = module_

    def __call__(self):
        self._setup_gpu_opt()
        self.module_.test_time_adapt(self.opt)

    def checkpoint(self):
        import os
        import submitit

        self.opt.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.opt)
        empty_trainer = type(self)(self.opt, self.module_)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_opt(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.opt.gpu = job_env.local_rank
        self.opt.rank = job_env.global_rank
        self.opt.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    script = sys.argv[1]
    assert (script in ["main", "baseline"])
    module_ = run_tta

    # if script == "main":
    #     module_ = main_classification
    # else:
    #     module_ = main_classification_baselines

    opt = parse_opt(module_)
    ensure_dir(opt.experiment_dir)
    executor = submitit.AutoExecutor(folder=opt.experiment_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = opt.ngpus
    nodes = opt.nodes
    timeout_min = opt.timeout

    partition = opt.partition
    kwopt = {}
    if opt.use_volta32:
        kwopt['slurm_constraint'] = 'volta32gb'
    if opt.comment:
        kwopt['slurm_comment'] = opt.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_qos="gpu_free",
        slurm_gres="gpu:{:d}".format(num_gpus_per_node)
        # slurm_signal_delay_s=120,
        # **kwopt
    )

    executor.update_parameters(name="SLAug")
    opt.dist_url = get_init_file().as_uri()
    trainer = Trainer(opt, module_)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {opt.experiment_dir}")


if __name__ == "__main__":
    main()
