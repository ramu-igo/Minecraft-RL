import json
import time

import numpy as np
import gym
import malmo.MalmoPython as MalmoPython

from .cmd_actions import ActionMgr


class BaseMalmoEnv(gym.Env):
    def __init__(self,
        width,
        height,
        agent_port,
        observer_port,
        task_actions,
        cam_angle=30,
        msec_per_tick=50,
        always_attack=False,
        interactive=False,
            ):
        super(BaseMalmoEnv, self).__init__()
        self.width = width
        self.height = height
        self.shape = (height, width, 3)
        self.msec_per_tick = msec_per_tick
        self.interactive = interactive

        self.agent_host = MalmoPython.AgentHost()
        self.pool = MalmoPython.ClientPool()
        self.pool.add(MalmoPython.ClientInfo('127.0.0.1', agent_port))

        if self.interactive:
            self.observe_host = MalmoPython.AgentHost()
            self.pool.add(MalmoPython.ClientInfo('127.0.0.1', observer_port))

        self.action_mgr = ActionMgr(task_actions, self.agent_host, cam_angle=cam_angle, always_attack=always_attack)
        self.action_space = gym.spaces.Discrete(self.action_mgr.n_action)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

        self.proc_loop_wait_sec = msec_per_tick * 0.2 / 1000
        self.mission_start_wait_sec = 0.3
        self.consume_state_wait_sec = msec_per_tick * 10 / 1000

    def _safe_start_mission(self, host, my_mission, role, exp_id='test', max_attempts=5):
        _mission_record = MalmoPython.MissionRecordSpec()
        _mission_record.recordRewards()
        _mission_record.recordObservations()

        attempts = 0
        while True:
            try:
                host.startMission(my_mission, self.pool, _mission_record, role, exp_id)
                break
            except MalmoPython.MissionException as e:
                error_code = e.details.errorCode
                if error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                    print('Server not quite ready yet - waiting...')
                    time.sleep(2)
                elif error_code == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                    print('Not enough available Minecraft instances running.')
                    attempts += 1
                    if attempts < max_attempts:
                        print('Will wait in case they are starting up.', max_attempts - attempts, 'attempts left.')
                        time.sleep(2)
                elif error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                    attempts += 1
                    if attempts < max_attempts:
                        time.sleep(2)
                    else:
                        print('Server not found - has the mission been started yet?')
                else:
                    print('Other error:', e.message)
                    print('Waiting will not help here - bailing immediately.')
                    raise
            if attempts == max_attempts:
                print('All chances used up - bailing now.')
                raise

    def _safe_wait_for_start(self, time_out=20):
        print('Waiting for the mission to start', end='')

        agent_hosts = [self.agent_host, self.observe_host]
        start_flags = [False for _ in agent_hosts]
        start_time = time.time()

        while not all(start_flags) and time.time() - start_time < time_out:
            states = [a.peekWorldState() for a in agent_hosts]
            start_flags = [w.has_mission_begun for w in states]
            errors = [e for w in states for e in w.errors]
            if len(errors) > 0:
                print('Errors waiting for mission start:')
                for e in errors:
                    print(e.text)
                print('Bailing now.')
                exit(1)
            time.sleep(0.1)
            print('.', end='')

        if time.time() - start_time >= time_out:
            print('Timed out while waiting for mission to start - bailing.')
            exit(1)

        print('\nMission has started.')

    def reset(self, malmo_xml_str):
        my_mission = MalmoPython.MissionSpec(malmo_xml_str, True)

        if self.interactive:
            self._safe_start_mission(self.agent_host, my_mission, role=0)
            self._safe_start_mission(self.observe_host, my_mission, role=1)
            self._safe_wait_for_start()

        else:
            self._safe_start_mission(self.agent_host, my_mission, role=0)
            world_state = self.agent_host.getWorldState()
            while not world_state.has_mission_begun:
                time.sleep(self.proc_loop_wait_sec)
                world_state = self.agent_host.getWorldState()

            time.sleep(self.mission_start_wait_sec)

        obs, *_ = self._process_state(False)
        self.action_mgr.reset()
        return obs

    def step(self, action):
        self.action_mgr.do_action(action)
        obs, reward, done, info = self._process_state()

        if done:
            self._consume_state()

        return obs, reward, done, info

    def close(self):
        self.quit_mission()
        self._process_state()
        self._consume_state()

    def quit_mission(self):
        self.agent_host.sendCommand('quit')
        if self.interactive:
            self.observe_host.sendCommand('quit')

    def _process_state(self, loop_max=500):
        reward = 0
        done = False
        info = {}
        obs = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        got_obs = False

        for _ in range(loop_max):
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_rewards_since_last_state > 0:
                reward += world_state.rewards[-1].getValue()
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
                obs = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(self.shape)
                got_obs = True
            if world_state.number_of_observations_since_last_state > 0:
                info = json.loads(world_state.observations[-1].text)

            done = not world_state.is_mission_running
            if done or got_obs:
                break

            time.sleep(self.proc_loop_wait_sec)

        return obs, reward, done, info

    def _consume_state(self, loop_max=1000):
        reward_flag = True
        reward = 0
        frame = None

        for _ in range(loop_max):
            time.sleep(self.consume_state_wait_sec)

            world_state = self.agent_host.getWorldState()
            if reward_flag and not (world_state.number_of_rewards_since_last_state > 0):
                reward_flag = False
            if reward_flag:
                reward = reward + world_state.rewards[-1].getValue()
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[-1]
            if not reward_flag:
                break

        return frame, reward
