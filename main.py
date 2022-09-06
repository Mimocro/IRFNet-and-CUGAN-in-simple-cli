import os, argparse, glob, gc
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="video_codec;hevc_nvenc"
import cv2
import numpy as np
import torch
import keyboard
import threading
import torchvision.transforms as F
from threading import Event
from PIL import Image
from subprocess import Popen, PIPE
from math import ceil

from models.CUGAN import RealWaifuUpScaler
from models.IFRNetLcastrated import Model as IFRNetL_Model
from models.IFRNetcastrated import Model as IFRNet_Model

def CUGAN(images, scale, upscaler):
    n = args.upscale / scale if args.upscale != 1 else 1
    for e in range(int(n)):
        for i in range(len(images)):
            images[i] = cv2.cvtColor(upscaler(cv2.cvtColor(np.array(images[i], dtype='uint8'), cv2.COLOR_RGB2BGR), args.upscale_tile, 1, 1), cv2.COLOR_BGR2RGB)  #[:, :, ::-1].copy() #[:, :, ::-1].copy() #input are brg, and then make it rgb back
    return images

#need to skip frames because its a stream
def skip_gpu_frames(decode_pipe, n_frames):
    for i in range(n_frames):
        _ = decode_pipe.stdout.read(w*h*3)
    
#thats really fine
last_i = None
last_raw_frame = None
def read_gpu(decode_pipe, i, w, h):
    global last_i
    global last_raw_frame
    if i != last_i:
        last_i = i
        raw_frame = decode_pipe.stdout.read(w*h*3)
        last_raw_frame = raw_frame
        frame =  np.frombuffer(raw_frame, dtype='uint8')
    else:
        frame = np.frombuffer(last_raw_frame, dtype='uint8')
    if frame.shape[0] == 0: return None
    else: 
        frame = frame.reshape((w, h, 3))
        frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
def read_frame(cap, n):
    #cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    res, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #frame[:,:,::-1].copy() #makes brg2rgb
    return frame #in rgb

def v_info(cap):
    #cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return(fps, frame_count)

def IFRNet(img0_np, img1_np):
    #gc.collect()
    #torch.cuda.empty_cache() #good old CUDA OOM error walktrough
    
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
    return ims #all frames in rgb
    
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


parser.add_argument("--no_segments", action='store_true', help='Do not process by segments')
parser.add_argument("--segment_l", default = 60, type=int)
parser.add_argument("-c", default = None, help='Path to txt file inside same folder as output, to continue process.')


parser.add_argument("--input_type", default="video", choices=['video', 'images'])
parser.add_argument("--base_fps", default=23.97602397602398, help='Used only if input is dir with images, by default it is 23.97602397602398 (equialent to 24000/1001)')
parser.add_argument("--images_ext", default='jpg', type=str)
parser.add_argument("--gpu_decode", action='store_true')
parser.add_argument("--gpu_encode", action='store_true')

parser.add_argument("--upscaler_model", default='pro-conservative', choices=["pro-conservative", "pro-denoise3x", "pro-no-denoise"], help='Model type, right file will be choised by upscale factor')
parser.add_argument("-m", "--mode", default="interpolate-upscale", choices=['upscale', 'interpolate', 'upscale-interpolate', 'interpolate-upscale'])
parser.add_argument("-u", "--upscale", default=2, type=int)
parser.add_argument("--upscale_tile", default=3, type=int)

parser.add_argument("--IFRNet_model", default='IFRNet', choices=['IFRNet', 'IFRNetL'])
parser.add_argument("-f", "--fps_multip", default=2, type=int)

args = parser.parse_args()


norm_path = os.path.normpath(os.path.abspath(args.input))
if args.output == None:
    output = norm_path+f' fps {fps} res {w}x{h}.mp4'
else:
    output = args.output
output = os.path.normpath(os.path.abspath(output))

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
    
if args.c != None:
    with open(os.path.normpath(args.c), 'r') as cfg:
        a = cfg.readlines()
        norm_path = a[0][:-2]
        output = a[1][:-2]
        args.input_type = a[2][:-2]
        args.images_ext = a[3][:-2]
        args.upscale = int(a[4])
        args.fps_multip = int(a[5])
        args.segment_l = int(a[6])
        n = int(a[7])
        
else:
    n = 0
    with open(output+'.txt', 'w') as cfg:
        for a in [norm_path, output, args.input_type, args.images_ext, args.upscale, args.fps_multip, args.segment_l, n]:
            cfg.write(fr"{a}"+' \n')
            
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
        

norm_path = os.path.normpath(os.path.abspath(norm_path))

output = os.path.normpath(os.path.abspath(output))
while os.path.isfile(output):
    output = f'{output[:-4]}_{i}{output[-4:]}'
    
if args.input_type == 'images': 
    w, h = frame[0].shape[0], frame[0].shape[1]
    fps_o = args.base_fps
    fps = args.fps_multip * args.base_fps if 'interpolate' in args.mode else args.base_fps
    frames = [file for file in glob.glob('{0}\\*.{1}'.format("\\".join(norm_path.split("\\")), args.images_ext))]
    frames_count = len(frames) 
else:
    cap_video = cv2.VideoCapture(norm_path)
    if args.gpu_decode: 
        decode_pipe = Popen(['ffmpeg', '-hide_banner' ,'-v', 'quiet', '-hwaccel', 'cuda', '-c:v', 'hevc_cuvid', '-i', norm_path, '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'], stdout=PIPE)
    w, h = read_frame(cap_video, 0).shape[0], read_frame(cap_video, 0).shape[1]
    fps_o, frames_count = v_info(cap_video)
    fps = args.fps_multip * fps_o if 'interpolate' in args.mode else fps_o
    
ffcmd = f'ffmpeg -hide_banner -v error -stats -y -f image2pipe -vcodec mjpeg -framerate {fps} -i - -c:v hevc_nvenc -rc vbr -cq 16 -preset slow -pix_fmt p010le -profile:v main10 -r {fps}' if args.gpu_encode else f'ffmpeg -hide_banner -v error -stats -y -f image2pipe -framerate {fps} -vcodec mjpeg -i - -c:v libx265 -crf 16 -preset slow -pix_fmt yuv420p10le -profile:v main10 -r {fps}'
num_n =  ceil(round(frames_count/fps_o) / args.segment_l) if not args.no_segments else 1
start = n if args.c != None else 0
last = round(start * args.segment_l * fps_o)

if start > 0 and args.gpu_decode:
    skip_gpu_frames(decode_pipe, last)

hotkey = 'ctrl+p'
print(f'\n\n\n PRESS "{hotkey}" to pause execution!')
running = Event()
running.set()
keyboard.add_hotkey(hotkey, handle_key_event, args=['down'])


frames_count_o = frames_count
print(frames_count_o)

if frames_count_o < args.segment_l*fps_o or args.no_segments:
    args.no_segments = True
    frames_count = frames_count_o
    
for nn in range(start, num_n):
    p = Popen(ffcmd.split(' ') + [output+str(nn)+output[-4:]], stdin=PIPE)
    frames_count = frames_count_o if frames_count_o - (args.segment_l*fps_o*nn) <= args.segment_l*fps_o or last > frames_count_o else round(args.segment_l * fps_o) +last
    if not args.no_segments:  print(f'Processing {nn+1} segment of {num_n}')
    for i in range(last, frames_count):
        if not running.is_set():
            print(f'Paused, press "{hotkey}" to continue')
            running.wait()
            print(f'Continued, press "{hotkey}" to pause')
            
        if args.input_type == 'images': 
            img0_np = cv2.imread(frames[i], mode='RGB')
        else: 
            if not args.gpu_decode: 
                img0_np = read_frame(cap_video, i)
            else: 
                img0_np = read_gpu(decode_pipe, i, w, h)
         
        ims = []



        if 'upscale' in args.mode and args.mode[:7] == 'upscale':
            img0_np = CUGAN([img0_np], clear_scale, upscaler)[0]
            
        if 'interpolate' not in args.mode:
            ims.append(Image.fromarray(img0_np))
        else:
            #print(i, i+1 > frames_count-1, '\n')
            if i+1 > frames_count-1:
                if 'upscale' in args.mode: Image.fromarray(CUGAN[img0_np]).save(p.stdin, 'JPEG')
                else: Image.fromarray(img0_np).save(p.stdin, 'JPEG')
                pass
            else: 
                if args.input_type == 'images': 
                    img1_np = cv2.imread(frames[i+1], mode='RGB')
                else: 
                    if not args.gpu_decode: 
                        img1_np = read_frame(cap_video, i+1)
                    else: 
                        img1_np = read_gpu(decode_pipe, i+1, w, h)
                        
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
    p.stdin.close()
    p.wait()
    
    last+=round(args.segment_l*fps_o)
    lines = open(output+'.txt', 'r').readlines()
    lines[-1] = str(nn)
    open(output+'.txt', 'w').writelines(lines)
    
if args.gpu_decode:
    decode_pipe.stdout.close()
    decode_pipe.wait()
    
if not args.no_segments:
    n = int(open(output+'.txt', 'r').readlines()[-1])
    a = ''
    for i in range(0, num_n):#last_segment+round(args.segment_l*fps_o), round(args.segment_l*fps_o)):
        a+= f"file '{output+str(i)+output[-4:]}'\n"
    vidlist = os.path.join(output, os.pardir) + '\\vidlist.txt'
    with open(vidlist, 'w')as vl:
        vl.write(a)
    os.system(f'ffmpeg -y -hide_banner  -v error -f concat -safe 0 -i vidlist.txt -c copy {output}')
    for i in range(0, num_n):
        os.remove(f'{output+str(i)+output[-4:]}')
else:
    os.rename(output+'0'+output[-4:], output)
    
if args.input_type == 'video':
  audio = f'{output}_audio.mp4'
  os.system(f'ffmpeg -y -hide_banner  -v error -i "{norm_path}" -vn -c copy -map 0:a "{audio}"')
else:
    audio = None
os.system(f'ffmpeg -y -hide_banner  -v error -i "{output}" -i "{audio}" -c copy "{output}+audio.mp4"')

i = 0

os.rename(f'{output}+audio.mp4', f'{output}')
os.remove(f'{output}')
os.remove(f'{vidlist}')
os.remove(f'{audio}')
