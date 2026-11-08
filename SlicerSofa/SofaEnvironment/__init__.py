import os
import sys
import platform

this_script_dir = os.path.dirname(os.path.abspath(__file__))

plugins = ['ArticulatedSystemPlugin',
           'MultiThreading',
           'SceneChecking',
           'SofaIGTLink',
           'SofaMatrix',
           'SofaPython3',
           'SofaValidation',
           'STLIB']

if platform.system() == "Windows":

    if os.path.isdir(this_script_dir + '/../../../../plugins'): #Install-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../../../'
        for plugin in plugins:
            sys.path = [this_script_dir + '/../../../../plugins/' + plugin + '/lib/python3/site-packages'] + sys.path
    else:                                         #Build-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../../../../Sofa-build'
        sys.path = [this_script_dir + '/../../../../../Sofa-build/lib/python3/site-packages'] + sys.path
# MacOS
elif platform.system() == "Darwin":
    if os.path.isdir(this_script_dir + '/../../python3/site-packages'): #Install-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../'
        sys.path = [this_script_dir + '/../../python3/site-packages'] + sys.path
    else:                                         #Build-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../../../../Sofa-build'
        sys.path = [this_script_dir + '/../../../../../Sofa-build/lib/python3/site-packages'] + sys.path

else:  #Linux

    if os.path.isdir(this_script_dir + '/../../Sofa'): #Install-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../Sofa'
        for plugin in plugins:
            sys.path = [this_script_dir + '/../../Sofa/plugins/' + plugin + '/lib/python3/site-packages'] + sys.path
    else:                                         #Build-tree
        os.environ['SOFA_ROOT'] = this_script_dir + '/../../../../../Sofa-build'
        sys.path = [this_script_dir + '/../../../../../Sofa-build/lib/python3/site-packages'] + sys.path

# Sofa, by default, will capture the exception handling. This is a workaround
# to keep the exception handling in PythonSlicer
defaultExecHook = sys.excepthook

import Sofa
import SofaRuntime

sys.excepthook = defaultExecHook

__all__ = ["Sofa", "SofaRuntime"]
