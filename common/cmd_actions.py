from collections import namedtuple


ACT_FORWARD = 'ACT_FORWARD'
ACT_BACK = 'ACT_BACK'
ACT_RIGHT = 'ACT_RIGHT'
ACT_LEFT = 'ACT_LEFT'
ACT_JUMP = 'ACT_JUMP'
ACT_ATTACK = 'ACT_ATTACK'
ACT_CAM_RIGHT = 'ACT_CAM_RIGHT'
ACT_CAM_LEFT = 'ACT_CAM_LEFT'
ACT_CAM_UP = 'ACT_CAM_UP'
ACT_CAM_DOWN = 'ACT_CAM_DOWN'
ACT_USE = 'ACT_USE'
ACT_USE_ON = 'ACT_USE_ON'
ACT_USE_OFF = 'ACT_USE_OFF'


class ActionMgr:

    def __init__(self, actions, agent_host, cam_angle=30, always_attack=False):
        """  
        - cam_angle: 一度のカメラ操作コマンドで向きを変える量
        - actions: タスクごとに設定する行動の選択肢. 
        (例)
          [
              [], # NO OP
              [ACT_FORWARD],
              [ACT_ATTACK],
              [ACT_FORWARD, ACT_JUMP], # 同時押し
              [ACT_CAM_RIGHT],
              [ACT_CAM_LEFT],
          ]       
        """
        self._actions = actions
        self.n_action = len(self._actions)
        self.always_attack = always_attack
        self.cur_doing_actions = set()
        self._agent_host = agent_host

        # malmoに送信するコマンドの準備
        # Continuousな挙動をするコマンドには'off'も指定。後で止めるときにoffの方を実行する(ボタンを離す事に相当)
        BaseAct = namedtuple('BaseAct', ['on', 'off'])
        self.commands = {
            ACT_FORWARD: BaseAct('forward 1', 'forward 0'),
            ACT_BACK: BaseAct('back 1', 'back 0'),
            ACT_RIGHT: BaseAct('right 1', 'right 0'),
            ACT_LEFT: BaseAct('left 1', 'left 0'),
            ACT_JUMP: BaseAct('jump 1', 'jump 0'),
            ACT_ATTACK: BaseAct('attack 1', 'attack 0'),
            ACT_CAM_RIGHT: BaseAct(f'moveMouse {cam_angle} 0', None),
            ACT_CAM_LEFT: BaseAct(f'moveMouse -{cam_angle} 0', None),
            ACT_CAM_UP: BaseAct(f'moveMouse 0 {cam_angle}', None),
            ACT_CAM_DOWN: BaseAct(f'moveMouse 0 -{cam_angle}', None),

            ACT_USE: BaseAct('use 1', 'use 0'),
            # 弓矢の場合、'use 1'で構え(チャージ開始)、'use 0'で打つ
            # 以下のようにON,OFF分けた方が、構えている間にも方向転換や低速移動はできるので自由度は高い
            ACT_USE_ON: BaseAct('use 1', None),
            ACT_USE_OFF: BaseAct('use 0', None),
        }

    def reset(self):
        self.cur_doing_actions.clear()
        if self.always_attack:
            # 最初に送信し、以降ずっと攻撃ON状態
            # ブロックはずっとON状態で壊し続ける
            # Mobに対してはON/OFFした時に攻撃するので、ずっとON状態にしても意味ない
            self._send_cmd(self.commands[ACT_ATTACK].on)

    def do_action(self, action_id):
        next_actions = self._actions[action_id]

        finish_acts = []
        for cur_act in self.cur_doing_actions:
            if cur_act not in next_actions:
                finish_acts.append(cur_act)
                self._send_cmd(self.commands[cur_act].off)
        for act in finish_acts:
            self.cur_doing_actions.remove(act)

        for next_act in next_actions:
            if next_act not in self.cur_doing_actions:
                self._send_cmd(self.commands[next_act].on)
                if self.commands[next_act].off is not None:
                    self.cur_doing_actions.add(next_act)

    def _send_cmd(self, cmd_str):
        self._agent_host.sendCommand(cmd_str)
