# Makes the environment, supports MetaWorld-v2 (mw) and ManiSkill3 (ms)
suites_to_try = []
try:
    import envs.metaworld as mw # MetaWorld-v2
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

