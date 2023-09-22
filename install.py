import os
import platform

import requests


def build_ebsynth():
    if os.path.exists('deps/ebsynth/bin/ebsynth'):
        print('Ebsynth has been built.')
        return

    os_str = platform.system()

    if os_str == 'Windows':
        print('Build Ebsynth Windows 64 bit.',
              'If you want to build for 32 bit, please modify install.py.')
        cmd = '.\\build-win64-cpu+cuda.bat'
        exe_file = 'deps/ebsynth/bin/ebsynth.exe'
    elif os_str == 'Linux':
        cmd = 'bash build-linux-cpu+cuda.sh'
        exe_file = 'deps/ebsynth/bin/ebsynth'
    elif os_str == 'Darwin':
        cmd = 'sh build-macos-cpu_only.sh'
        exe_file = 'deps/ebsynth/bin/ebsynth.app'
    else:
        print('Cannot recognize OS. Ebsynth installation stopped.')
        return

    os.chdir('deps/ebsynth')
    print(cmd)
    os.system(cmd)
    os.chdir('../..')
    if os.path.exists(exe_file):
        print('Ebsynth installed successfully.')
    else:
        print('Failed to install Ebsynth.')


def download(url, dir, name=None):
    os.makedirs(dir, exist_ok=True)
    if name is None:
        name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Install {name} ...')
        open(path, 'wb').write(requests.get(url).content)
        print('Install successfully.')


def download_gmflow_ckpt():
    url = ('https://huggingface.co/PKUWilliamYang/Rerender/'
           'resolve/main/models/gmflow_sintel-0c07dcb3.pth')
    download(url, 'models')


def download_controlnet_canny():
    url = ('https://huggingface.co/lllyasviel/ControlNet/'
           'resolve/main/models/control_sd15_canny.pth')
    download(url, 'models')


def download_controlnet_hed():
    url = ('https://huggingface.co/lllyasviel/ControlNet/'
           'resolve/main/models/control_sd15_hed.pth')
    download(url, 'models')


def download_vae():
    url = ('https://huggingface.co/stabilityai/sd-vae-ft-mse-original'
           '/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt')
    download(url, 'models')


build_ebsynth()
download_gmflow_ckpt()
download_controlnet_canny()
download_controlnet_hed()
download_vae()
