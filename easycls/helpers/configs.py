from configparser import ConfigParser

# https://www.liaoxuefeng.com/wiki/1016959663602400/1018490750237280
# config default and overide

def parse_cfgs(path):
    cfgparser = ConfigParser()
    cfgparser.read(path)

    # sections = cfgparser.sections()
    # # print(cfgparser._sections)
    # for sect in sections:
    #     print(cfgparser._sections[sect])

    # return cfgparser._sections
    return cfgparser