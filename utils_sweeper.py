#!/home/yuandong/anaconda/bin
import argparse
import imp

def param_to_arg(param, env_vars=None):
    ''' Turn param to "--key value ", if value is bool, then
        it will switch on/off the key depending on the value
        env_vars exluded.
    '''
    arg = ""
    if isinstance(param, dict):
        param = [ (k, v) for k, v in param.items() ]
    for (k, v) in param:
        if env_vars is not None and k in env_vars: continue
        if k != "--":
            this_arg = " --%s" % k
            if not isinstance(v, bool):
                if isinstance(v, str) and (":" in v or ";" in v or "|" in v):
                    v = "\"" + v + "\""
                this_arg += " " + str(v)
                arg += this_arg
            else:
                if v: arg += this_arg
        else:
            arg += " --%s" % v
    return arg

def param_to_env(param, env_vars):
    if isinstance(env_vars, str):
        return env_vars
    else:
        if isinstance(param, dict):
            param = [ (k, v) for k, v in param.items() ]
        return " ".join("%s=%s" % (k, str(v)) for (k, v) in param if k in env_vars)


def iter_params(params):
    if isinstance(params, list):
        # List: just go through them one by one.
        for param in params:
            yield from iter_params(param)
    elif isinstance(params, dict):
        # remove the first key and yield on the rest
        if len(params) == 0:
            yield []
        else:
            # For each value of the first key,
            #   (key, value) + iter_params(other param)
            first_key = sorted(params.keys())[0]
            dup_params = dict(params)
            del dup_params[first_key]

            if first_key.startswith("__sub"):
                for rest_instance in iter_params(dup_params):
                    for first_instance in iter_params(params[first_key]):
                        # print("__sub: First: " + str(first_instance))
                        # print("__sub: Rest: " + str(rest_instance))
                        yield first_instance + rest_instance
            else:
                for rest_instance in iter_params(dup_params):
                    # print("Rest_instance = " + str(rest_instance))
                    for first_val_instance in iter_params(params[first_key]):
                        # print("Key " + first_key + " val: " + str(first_val_instance))
                        yield [(first_key, first_val_instance)] + rest_instance
    else:
        yield params


def list2dict(l):
    return { p[0] : p[1] for p in l }


# Load parameter files and generate args.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, help="Parameter file")
    parser.add_argument("--idx", type=int, default=None)

    args = parser.parse_args()

    custom_params = imp.load_source("custom_params", args.param_file)
    params = custom_params.params

    params = list(iter_params(params))

    if args.idx is None:
        for param in params:
            arg = param_to_arg(param)
            print(arg)
    else:
        print(param_to_arg(params[args.idx]))

