import os

from common.base_malmo_env import BaseMalmoEnv
from common.cmd_actions import *
from common.env_wrappers import *

from mission_xml.xml_helper import load_mission_xml


AGENT_PORT = 10000
OBSERVER_PORT = 10001

WORLD_DATA = './world/field_zombie_1'
MISSION_XML = './mission_xml/zombie_battle.xml'

REWARD_FOR_VICTORY = 10
REWARD_FOR_ATTACK = 3
REWARD_FOR_DAMAGE = -0.5
REWARD_FOR_DEAD = -3

AGENT_ACTIONS = [
    [ACT_FORWARD],
    [ACT_RIGHT],
    [ACT_LEFT],
    [ACT_BACK],
    [ACT_ATTACK],
    [ACT_CAM_RIGHT],
    [ACT_CAM_LEFT],
]

CAM_ANGLE = 30

MODEL_INPUT_W = 84
MODEL_INPUT_H = 84

AGENT_IMG_W = 256
AGENT_IMG_H = 256

OBSERVER_CONF = {
    'x': -2353,
    'y': 9,
    'z': -248,
    'yaw': 0,
    'pitch': 34,
    'img_w': 800,
    'img_h': 600,
}


class ZombieBattleEnv(BaseMalmoEnv):
    def __init__(self, 
        width, height, agent_port, observer_port,
        msec_per_tick=50, interactive=False, 
            ):
        self.world_data_dir = os.path.abspath(WORLD_DATA)
        assert os.path.exists(self.world_data_dir), 'world data not found.'

        super().__init__(
            width=width,
            height=height,
            agent_port=agent_port,
            observer_port=observer_port,
            task_actions=AGENT_ACTIONS,
            cam_angle=CAM_ANGLE,
            msec_per_tick=msec_per_tick,
            interactive=interactive,
        )

        self.reset_mission_params()

    def reset_mission_params(self):
        self.cur_life = None
        self.cur_num_zombie = None

    def reset(self):
        self.reset_mission_params()

        xml = load_mission_xml(MISSION_XML,
                               width=self.width,
                               height=self.height,
                               world_data=self.world_data_dir,
                               obs_conf=OBSERVER_CONF,
                               reward_for_attack=REWARD_FOR_ATTACK,
                               msec_per_tick=self.msec_per_tick,
                               interactive=self.interactive
                               )

        return super().reset(xml)

    def step(self, action):
        """ info(malmoからの観測情報)を参照し、rewardの設定やミッション終了の判断を行う """
        obs, reward, done, info = super().step(action)

        if 'entities' in info:
            entities = info['entities']
            num_zombie = 0
            for ent in entities:
                # count zombie
                if ent['name'] == 'Zombie':
                    num_zombie += 1

            if self.cur_num_zombie is not None:
                if self.cur_num_zombie > 0 and num_zombie == 0:
                    # ゾンビを全滅させた時
                    reward += REWARD_FOR_VICTORY
                    self.quit_mission()
            self.cur_num_zombie = num_zombie

        if 'Life' in info:
            life = info['Life']
            if self.cur_life is not None:
                if self.cur_life > 0 and life == 0:
                    # 死んだとき
                    reward += REWARD_FOR_DEAD
                    if self.interactive:
                        self.quit_mission()
                elif life < self.cur_life:
                    # ダメージを受けた時
                    reward += REWARD_FOR_DAMAGE
            self.cur_life = life

        return obs, reward, done, info


def build_env(msec_per_tick=50, interactive=False):
    env = ZombieBattleEnv(
        width=AGENT_IMG_W,
        height=AGENT_IMG_H,
        agent_port=AGENT_PORT,
        observer_port=OBSERVER_PORT,
        msec_per_tick=msec_per_tick,
        interactive=interactive,
    )
    env = WarpFrame(env, width=MODEL_INPUT_W, height=MODEL_INPUT_H, grayscale=False)
    env = FrameSkip(env, skip=4)
    return env
