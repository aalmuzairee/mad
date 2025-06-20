# Makes the environment, supports Meta-World (mw) and ManiSkill3 (ms)
suites_to_try = []
try:
    import envs.metaworld as mw # Meta-World
    suites_to_try.append(mw)
except:
    pass
try:
    import envs.maniskill as ms # ManiSkill3
    suites_to_try.append(ms)
except:
    pass

def make(cfg):
    env = None
    for each_suite in suites_to_try:
        try:
            env = each_suite.make(cfg)
            break
        except:
            pass

    if env == None:
        raise KeyError("None of the envs matched, please check your config")

    return env


# Meta-World tasks
mw_tasks = ['basketball',
            'hammer',
            'peg-insert-side',
            'soccer',
            'sweep-into',
            'assembly',
            'hand-insert',
            'pick-out-of-hole',
            'pick-place',
            'push',
            'shelf-place',
            'disassemble',
            'stick-pull',
            'stick-push',
            'pick-place-wall']

# ManiSkill3 tasks
ms_tasks = ['PokeCube', 
            'PlaceSphere', 
            'PickCube', 
            'PushCube', 
            'PullCube']
