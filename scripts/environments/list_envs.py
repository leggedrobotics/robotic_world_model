"""列出当前扩展注册的任务环境。

启动 Isaac Sim 后扫描 gym registry，并输出任务名/入口点/配置项。
"""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


# 其余逻辑用于遍历并格式化打印任务表

import gymnasium as gym
from prettytable import PrettyTable

# Import extensions to set up environment tasks
import mbrl.tasks  # noqa: F401


def main():
    """Print all environments registered in `isaac.lab_demo` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab Template Extension"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "Template-" in task_spec.id:
            # add details to table
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs["env_cfg_entry_point"],
                ]
            )
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
