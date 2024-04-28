import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(args):
    # An instance of the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=1)
    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cuda().eval()

    # An example input
    x = torch.rand(1, 3, 64, 64).cuda()
    # Export the model
    # with torch.no_grad():
    #     torch_out = torch.onnx._export(model, x, args.output, opset_version=11, export_params=True)
    save_onnx_path = args.output

    # print(x.shape)
    if args.use_fp16:
        x = x.half()
        model = model.half()
        print("use fp16")

    if args.dynamic_axes:
        print("use dynamic shape")
        dynamic_axes = {'input1' : {2: 'in_height', 3: 'in_width'}, 'output' : {2 : 'out_height', 3: 'out_width'}}
        torch.onnx.export(model, x, save_onnx_path, export_params=True, opset_version=17, do_constant_folding=True, input_names=['input1'], output_names=['output'], dynamic_axes=dynamic_axes, verbose=False)
    else:
        with torch.no_grad():
            torch.onnx.export(model, x, save_onnx_path, export_params=True, opset_version=17, do_constant_folding=True, input_names=['input1'],output_names=['output'],verbose=True)

    # print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='experiments/pretrained_models/RealESRGAN_x4plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan-x4.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    parser.add_argument('--use_fp16', action='store_true', help='Use fp16')
    parser.add_argument('--dynamic_axes', action='store_true', help='Use dynamic shape')
    args = parser.parse_args()

    main(args)
