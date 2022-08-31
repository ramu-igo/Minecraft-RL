from pathlib import Path


def gen_observer_section(
        x, y, z, yaw, pitch,
        img_w=900, img_h=600, name="Observer", is_creative=True):

    templ = '''
      <AgentSection mode="{MODE}">
        <Name>{NAME}</Name>
        <AgentStart>
          <Placement x="{X}" y="{Y}" z="{Z}" yaw="{YAW}" pitch="{PITCH}" />
        </AgentStart>
        <AgentHandlers>
          <MissionQuitCommands/>
          <VideoProducer>
            <Width>{WIDTH}</Width>
            <Height>{HEIGHT}</Height>
          </VideoProducer>
        </AgentHandlers>
      </AgentSection>
    '''

    mode = "Creative" if is_creative else "Survival"
    ret = templ.format(MODE=mode, NAME=name, X=x, Y=y, Z=z, YAW=yaw, PITCH=pitch,
        WIDTH=img_w, HEIGHT=img_h)
    return ret


def gen_quit_timeup(timelimit_sec=90, description=""):

    templ = '''
      <ServerQuitFromTimeUp description="{DESC}" timeLimitMs="{LIMIT_MSEC}" />
    '''

    limit_msec = timelimit_sec * 1000
    ret = templ.format(DESC=description, LIMIT_MSEC=limit_msec)
    return ret


def load_mission_xml(xml_path, width, height, world_data, 
                     obs_conf, reward_for_attack,
                     msec_per_tick=50, force_reset=True, interactive=False):
    xml = Path(xml_path).read_text()

    _reset = "true" if force_reset else "false"
    _quit_timeup = gen_quit_timeup()

    _observer = ""
    if interactive:
        _observer = gen_observer_section(
            x=obs_conf["x"], y=obs_conf["y"], z=obs_conf["z"],
            yaw=obs_conf["yaw"], pitch=obs_conf["pitch"],
            img_w=obs_conf["img_w"], img_h=obs_conf["img_h"])

        # interactiveの時はdemo用として、<ServerQuitFromTimeUp>を消す
        # (画面にカウントダウンが表示されなくなり、見やすくなる)
        _quit_timeup = ""

    xml = xml.format(
                MS_TICK=msec_per_tick, FORCE_RESET=_reset, QUIT_TIMEUP=_quit_timeup,
                WORLD_DATA=world_data, REWARD_ATTACK=reward_for_attack,
                WIDTH=width, HEIGHT=height, OBSERVER=_observer)

    return xml

