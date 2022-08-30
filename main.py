import os, argparse, glob, gc
import cv2
import numpy as np
import torch
import keyboard
import threading
import torchvision.transforms as F
from threading import Event
from PIL import Image
from subprocess import Popen, PIPE

from models.CUGAN import RealWaifuUpScaler
from models.IFRNetLcastrated import Model as IFRNetL_Model
from models.IFRNetcastrated import Model as IFRNet_Model









#why so slow...
def CUGAN(images, scale, upscaler):
    n = args.upscale / scale if args.upscale != 1 else 1
    for e in range(int(n)):
        for i in range(len(images)):
            images[i] = upscaler(np.array(images[i], dtype='uint8')[:, :, ::-1].copy(), args.upscale_tile, 1, 1)

    return images

def read_frame(filename, n):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    res, frame = cap.read()
    frame = frame[:,:,::-1].copy()
    return frame

def v_info(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return(fps, frame_count)

def IFRNet(img0_np, img1_np):
    gc.collect()
    torch.cuda.empty_cache() #good old CUDA OOM error walktrough
    
    with torch.no_grad():
        img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).half() / 255.0).unsqueeze(0).cuda()
        img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).half() / 255.0).unsqueeze(0).cuda()
        
        emba = []
        for i in range(1, args.fps_multip):
            emba.append(torch.tensor(i/args.fps_multip).view(1, 1, 1, 1).half().cuda())
        #imgt_pred = model.inference(img0, img1, embt)
        img0_ = [img0]
        img1_ = [img1]
        img0_ = torch.cat([x for x in img0_ for i in range(args.fps_multip-1)], 0)
        img1_ = torch.cat([x for x in img1_ for i in range(args.fps_multip-1)], 0)
        embt = torch.cat([x for x in emba], 0)
        imgt_pred = model_ifrnet.inference(img0_, img1_, embt)

        ims = []
        for i in range(args.fps_multip-1):
            ims.append(F.ToPILImage()(imgt_pred[i]))
        ims.append(Image.fromarray(img1_np))
    return ims
    
def handle_key_event(event):
    gc.collect()
    torch.cuda.empty_cache()
    if event == 'down':
        if running.is_set():
            running.clear()
        else:
            running.set()



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help='Path to input video file or dir with images (use --input_type to choose type of input)')
parser.add_argument("-o", "--output", default = None)

parser.add_argument("--input_type", default="video", choices=['video', 'images'])
parser.add_argument("--base_fps", default=23.97602397602398, help='Used only if input is dir with images, by default it is 23.97602397602398 (equialent to 24000/1001)')
parser.add_argument("--images_ext", default='jpg', type=str)

parser.add_argument("--upscaler_model", default='pro-conservative', choices=["pro-conservative", "pro-denoise3x", "pro-no-denoise"], help='Model type, right file will be choised by upscale factor')
parser.add_argument("-m", "--mode", default="interpolate-upscale", choices=['upscale', 'interpolate', 'upscale-interpolate', 'interpolate-upscale'])
parser.add_argument("-u", "--upscale", default=2, type=int)
parser.add_argument("--upscale_tile", default=3, type=int)

parser.add_argument("--IFRNet_model", default='IFRNet', choices=['IFRNet', 'IFRNetL'])
parser.add_argument("-f", "--fps_multip", default=2, type=int)

args = parser.parse_args()


norm_path = os.path.normpath(args.input)

clear_scale = 3 if args.upscale % 3 == 0 else 2
upscale_model = f'./CUGAN/{args.upscaler_model}-up{clear_scale}x.pth'
if args.upscale % 2 != 0 and args.upscale % 3 != 0 and args.upscale != 1:
    raise ValueError('Upscaling factor must be divabable to 2 or 3!')
if os.path.isfile(norm_path) and args.input_type != 'video':
    raise ValueError('If your pass video file, check path and option --input_type!')
if os.path.isdir(norm_path) and args.input_type == 'video':
    raise ValueError('If your pass images folder,  check path and option --input_type!')
if not os.path.isfile(norm_path) and not os.path.isdir(norm_path):
    raise ValueError('No such file or dirrectory!')
    




device = "cuda"
with torch.no_grad():
    if 'interpolate' in args.mode:
        if args.IFRNet_model == 'IFRNet':
            model_ifrnet = IFRNet_Model().cuda().eval()
            model_ifrnet.load_state_dict(torch.load('./IFRNet/IFRNet_Vimeo90K.pth')) #load large model
        if args.IFRNet_model == 'IFRNetL':
            model_ifrnet = IFRNetL_Model().cuda().eval()
            model_ifrnet.load_state_dict(torch.load('./IFRNet_L/IFRNet_L_Vimeo90K.pth')) #load large model
        model_ifrnet.half() #and half to save vram
        
    if 'upscale' in args.mode:
        upscaler = RealWaifuUpScaler(clear_scale, upscale_model, half=True, device=device)
        

if args.input_type == 'images': 
    w, h = frame[0].shape[0], frame[0].shape[1]
    fps = args.fps_multip * base_fps if 'interpolate' in args.mode else base_fps
    frames = [file for file in glob.glob('{0}\\*.{1}'.format("\\".join(norm_path.split("\\")), args.images_ext))]
    frames_count = len(frames) 
else:
    w, h = read_frame(norm_path, 0).shape[0], read_frame(norm_path, 0).shape[1]
    fps, frames_count = v_info(norm_path)
    fps = args.fps_multip * fps if 'interpolate' in args.mode else fps

if args.output == None:
    output = norm_path+f' fps {fps} res {w}x{h}.mp4'
else:
    output = args.output
output = os.path.normpath(output)

p = Popen(['ffmpeg', '-hide_banner' ,'-v', 'error', '-stats', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', f'{fps}', '-i', '-', '-c:v', 'libx265', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p10le','-profile:v', 'main10', '-r', f'{fps}', f'{output}'], stdin=PIPE)


hotkey = 'ctrl+p'
print(f'\n\n\n PRESS "{hotkey}" to pause execution!')
running = Event()
running.set()
keyboard.add_hotkey(hotkey, handle_key_event, args=['down'])

#keyboard.hook_key(hotkey, handle_key_event)


for i in range(frames_count):
    try:
        if not running.is_set():
            print(f'Paused, press "{hotkey}" to continue')
            running.wait()
            print(f'Continued, press "{hotkey}" to pause')

        img0_np = cv2.imread(frames[i], mode='RGB') if args.input_type == 'images' else read_frame(norm_path, i)

        
        ims = []
        #t0 = time.time()
        if 'upscale' in args.mode and args.mode[:7] == 'upscale':
            img0_np = CUGAN([img0_np], clear_scale, upscaler)[0]
            
        #total02 = t1-t0
        #print('total02', total02)
        
        if 'interpolate' not in args.mode:
            ims.append(Image.fromarray(img0_np))
        else:
            if i+1 > frames_count:
                ims.append(Image.fromarray(img0_np))
            else: 
                img1_np = cv2.imread(frames[i+1], mode='RGB') if args.input_type == 'images' else read_frame(norm_path, i+1)
                if 'upscale' in args.mode and args.mode[:7] == 'upscale':
                    img1_np = CUGAN([img1_np], clear_scale, upscaler)[0]
                ims = IFRNet(img0_np, img1_np)
                if 'upscale' in args.mode and args.mode[:7] != 'upscale':
                    ims =  CUGAN(ims, clear_scale, upscaler)
                    for i in range(len(ims)):
                        ims[i] = Image.fromarray(ims[i])
                if i==0: 
                    if 'upscale' in args.mode and args.mode[:7] != 'upscale':
                        ims.insert(0, Image.fromarray(CUGAN([img0_np], clear_scale, upscaler)[0]))
                    else:
                        ims.insert(0, Image.fromarray(img0_np))
            
            
            
        for frame in ims:
            frame.save(p.stdin, 'JPEG')
            
            
    except KeyboardInterrupt:
        print('Interrupted')
        del model_ifrnet
        try:
            for frame in ims:
                frame.save(p.stdin, 'JPEG')
            break
        except:
            break
p.stdin.close()
p.wait()

if args.input_type == 'video':
   audio = f'{output}.mp4'
   os.system(f'ffmpeg -y -hide_banner -i "{norm_path}" -vn -c copy -map 0:a "{audio}"')
else:
   audio = None
os.system(f'ffmpeg -y -i "{output}" -i "{audio}" -c copy "{output}+audio.mp4"')
os.remove(f'{output}')
os.remove(f'{audio}')
os.rename(f'{output}+audio.mp4', f'{output}')
