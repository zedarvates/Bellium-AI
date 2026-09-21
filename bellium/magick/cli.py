from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from bellium.knn._image import shape
from bellium.magick.animation import assemble_spritesheet, slice_spritesheet
from bellium.magick.auto_orient import auto_orient
from bellium.magick.fastimage import fast_info
from bellium.magick.geometry_parser import parse_geometry
from bellium.magick.hashes import compute_ahash, compute_dhash, compute_phash, hash_to_hex
from bellium.magick.io import load_image, save_image
from bellium.magick.mogrify import mogrify_batch
from bellium.specialists.magick import magick_process

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='bellium-magick',
        description='Pure-Python / k-NN / Nano-NN ImageMagick replacement suite',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # info
    p_info = subparsers.add_parser('info', help='Display image metadata and color statistics')
    p_info.add_argument('input', help='Path to input image')
    
    # fastimage
    p_fast = subparsers.add_parser('fastimage', help='Instant header sniffing without decoding full image (O(1) memory)')
    p_fast.add_argument('input', help='Path to input image file')
    
    # phash
    p_phash = subparsers.add_parser('phash', help='Compute perceptual, difference, and average 64-bit hashes')
    p_phash.add_argument('input', help='Path to input image')
    
    # auto-orient
    p_orient = subparsers.add_parser('auto-orient', help='Automatically orient image based on EXIF orientation tag')
    p_orient.add_argument('input', help='Input image path')
    p_orient.add_argument('output', help='Output image path')
    
    # resize
    p_resize = subparsers.add_parser('resize', help='Resize image using nano-NN edge, k-NN or deterministic baselines')
    p_resize.add_argument('input', help='Input image path')
    p_resize.add_argument('output', help='Output image path')
    p_resize.add_argument('--width', type=int, help='Target width')
    p_resize.add_argument('--height', type=int, help='Target height')
    p_resize.add_argument('--geometry', help='ImageMagick geometry spec (e.g. 800x600, 50%%, 800x600^, 800x600#)')
    p_resize.add_argument('--method', choices=['nano_edge', 'knn', 'bilinear', 'nearest', 'box'], default='nano_edge')
    
    # quantize
    p_quant = subparsers.add_parser('quantize', help='Quantize palette with nano-NN dither arbiter or k-NN mapping')
    p_quant.add_argument('input', help='Input image path')
    p_quant.add_argument('output', help='Output image path')
    p_quant.add_argument('--colors', type=int, default=16, help='Number of palette colors')
    p_quant.add_argument('--dither', choices=['nano_tone', 'knn', 'floyd_steinberg', 'none'], default='nano_tone')
    
    # filter
    p_filt = subparsers.add_parser('filter', help='Apply spatial convolutions or adaptive nano-NN filtering')
    p_filt.add_argument('input', help='Input image path')
    p_filt.add_argument('output', help='Output image path')
    p_filt.add_argument('--type', dest='filter_type', choices=['auto', 'nano_adaptive', 'sharpen', 'blur', 'edge', 'emboss'], default='auto')
    p_filt.add_argument('--strength', type=float, default=1.0, help='Filter strength')
    p_filt.add_argument('--radius', type=int, default=1, help='Filter radius')
    p_filt.add_argument('--sigma', type=float, default=1.0, help='Gaussian sigma')
    
    # tone
    p_tone = subparsers.add_parser('tone', help='Adjust tones via micro-NN curve or histogram stretch')
    p_tone.add_argument('input', help='Input image path')
    p_tone.add_argument('output', help='Output image path')
    p_tone.add_argument('--mode', choices=['micro_nn', 'stretch', 'equalize', 'gamma'], default='micro_nn')
    p_tone.add_argument('--target-luma', type=float, default=128.0, help='Target luma for gamma')
    
    # threshold
    p_thresh = subparsers.add_parser('threshold', help='Binarize image with k-NN strategy selection or Otsu')
    p_thresh.add_argument('input', help='Input image path')
    p_thresh.add_argument('output', help='Output image path')
    p_thresh.add_argument('--mode', choices=['knn', 'otsu'], default='knn')
    
    # morphology
    p_morph = subparsers.add_parser('morphology', help='Mathematical morphology: dilate, erode, open, close')
    p_morph.add_argument('input', help='Input image path')
    p_morph.add_argument('output', help='Output image path')
    p_morph.add_argument('--op', choices=['dilate', 'erode', 'open', 'close'], default='dilate')
    p_morph.add_argument('--radius', type=int, default=1)
    
    # compare
    p_comp = subparsers.add_parser('compare', help='Compute exact RMSE, PSNR, SSIM and optional visual diff')
    p_comp.add_argument('image_a', help='Path to reference image')
    p_comp.add_argument('image_b', help='Path to distorted/test image')
    p_comp.add_argument('--diff-output', help='Optional path to write diff image')
    
    # transform
    p_trans = subparsers.add_parser('transform', help='Rotate, flip, pad or trim image')
    p_trans.add_argument('input', help='Input image path')
    p_trans.add_argument('output', help='Output image path')
    p_trans.add_argument('--op', required=True, choices=['flip_h', 'flip_v', 'rotate90', 'rotate180', 'rotate270', 'trim', 'pad'])
    p_trans.add_argument('--target-width', type=int, help='Target width for pad')
    p_trans.add_argument('--target-height', type=int, help='Target height for pad')
    
    # composite
    p_comp_layer = subparsers.add_parser('composite', help='Layer composite with blend modes')
    p_comp_layer.add_argument('input', help='Base image path')
    p_comp_layer.add_argument('overlay', help='Overlay image path')
    p_comp_layer.add_argument('output', help='Output image path')
    p_comp_layer.add_argument('--mode', choices=['over', 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'difference', 'add', 'subtract', 'color_dodge', 'soft_light'], default='over')
    p_comp_layer.add_argument('--opacity', type=float, default=1.0)
    
    # channel
    p_chan = subparsers.add_parser('channel', help='Extract or combine color channels')
    p_chan.add_argument('input', help='Input image path')
    p_chan.add_argument('output', help='Output image path')
    p_chan.add_argument('--action', choices=['extract', 'combine'], default='extract')
    p_chan.add_argument('--channel', choices=['r', 'g', 'b', 'luma'], default='luma')
    p_chan.add_argument('--green', help='Green channel image for combine')
    p_chan.add_argument('--blue', help='Blue channel image for combine')
    
    # montage
    p_mont = subparsers.add_parser('montage', help='Create contact sheet or sprite grid collage')
    p_mont.add_argument('inputs', nargs='+', help='List of input image paths')
    p_mont.add_argument('-o', '--output', required=True, help='Output image path')
    p_mont.add_argument('--columns', type=int, help='Number of grid columns')
    p_mont.add_argument('--rows', type=int, help='Number of grid rows')
    p_mont.add_argument('--padding', type=int, default=2, help='Padding between tiles')
    
    p_vec = subparsers.add_parser('vectorize', help='Convert flat-color raster art to SVG via k-NN region routing')
    p_vec.add_argument('input', help='Input image path')
    p_vec.add_argument('output', help='Output .svg path')
    p_vec.add_argument('--colors', type=int, default=8, help='Palette size before tracing')
    p_vec.add_argument('--min-area', type=int, default=4, help='Drop speck regions smaller than this')
    p_vec.add_argument('--epsilon', type=float, default=0.6, help='Path simplification tolerance')
    
    p_mog = subparsers.add_parser('mogrify', help='Batch apply image transformations across multiple files')
    p_mog.add_argument('inputs', nargs='+', help='List of files to process')
    p_mog.add_argument('--output-dir', help='Optional directory to write outputs')
    p_mog.add_argument('--format', help='Target output format (e.g. ppm, json, png)')
    p_mog.add_argument('--resize', help='Resize geometry such as 50 percent or 800x600')
    p_mog.add_argument('--filter', dest='filter_type', choices=['auto', 'nano_adaptive', 'sharpen', 'blur', 'edge', 'emboss'])
    p_mog.add_argument('--tone', dest='tone_mode', choices=['micro_nn', 'stretch', 'equalize', 'gamma'])
    p_mog.add_argument('--quantize', type=int, dest='quantize_colors', help='Quantize to N colors')
    p_mog.add_argument('--auto-orient', action='store_true', help='Auto-orient based on EXIF tag')
    
    p_slice = subparsers.add_parser('spritesheet-slice', help='Slice a spritesheet into individual frames')
    p_slice.add_argument('input', help='Input spritesheet path')
    p_slice.add_argument('output_dir', help='Output directory for frame images')
    p_slice.add_argument('--frame-width', type=int, required=True, help='Frame width in pixels')
    p_slice.add_argument('--frame-height', type=int, required=True, help='Frame height in pixels')
    p_slice.add_argument('--format', default='ppm', help='Output frame format (default: ppm)')
    
    p_ass = subparsers.add_parser('spritesheet-assemble', help='Assemble frames into a spritesheet strip or grid')
    p_ass.add_argument('inputs', nargs='+', help='List of frame images')
    p_ass.add_argument('-o', '--output', required=True, help='Output spritesheet path')
    p_ass.add_argument('--direction', choices=['horizontal', 'vertical', 'grid'], default='horizontal')
    p_ass.add_argument('--columns', type=int, help='Columns for grid mode')
    
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    if args.command == 'info':
        img = load_image(args.input)
        h, w = shape(img)
        pixels = [px for row in img for px in row]
        lumas = [(2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0 for p in pixels]
        info = {
            'path': str(args.input),
            'dimensions': f'{w}x{h}',
            'width': w,
            'height': h,
            'mean_luma': round(sum(lumas) / len(lumas), 2),
            'min_luma': round(min(lumas), 2),
            'max_luma': round(max(lumas), 2),
        }
        print(json.dumps(info, indent=2))
        return 0
        
    elif args.command == 'fastimage':
        fi = fast_info(args.input)
        print(json.dumps({
            'path': str(args.input),
            'format': fi.format,
            'width': fi.width,
            'height': fi.height,
            'channels': fi.channels,
            'bit_depth': fi.bit_depth,
            'orientation': fi.orientation,
            'is_animated': fi.is_animated,
            'aspect_ratio': fi.aspect_ratio,
            'bytes_inspected': fi.bytes_read,
        }, indent=2))
        return 0
        
    elif args.command == 'phash':
        img = load_image(args.input)
        ahash_val = compute_ahash(img)
        dhash_val = compute_dhash(img)
        phash_val = compute_phash(img)
        print(json.dumps({
            'path': str(args.input),
            'ahash': hash_to_hex(ahash_val),
            'dhash': hash_to_hex(dhash_val),
            'phash': hash_to_hex(phash_val),
        }, indent=2))
        return 0
        
    elif args.command == 'auto-orient':
        fi = fast_info(args.input)
        img = load_image(args.input)
        oriented = auto_orient(img, fi.orientation)
        save_image(oriented, args.output)
        print(f'Auto-oriented image (EXIF tag {fi.orientation}) and saved to {args.output}')
        return 0
        
    elif args.command == 'compare':
        img_a = load_image(args.image_a)
        img_b = load_image(args.image_b)
        res = magick_process({'image': img_a, 'image_b': img_b, 'operation': 'compare'})
        metrics = res.output['metrics']
        print(json.dumps(metrics, indent=2))
        if args.diff_output and 'diff_image' in res.output:
            save_image(res.output['diff_image'], args.diff_output)
        return 0
        
    elif args.command == 'montage':
        imgs = [load_image(src) for src in args.inputs]
        res = magick_process({'image': imgs[0], 'images': imgs, 'operation': 'montage', 'columns': args.columns, 'rows': args.rows, 'padding': args.padding})
        save_image(res.output['image'], args.output)
        print(f'Created montage with {len(imgs)} images -> {args.output}')
        return 0
        
    elif args.command == 'vectorize':
        img = load_image(args.input)
        res = magick_process({
            'image': img,
            'operation': 'vectorize',
            'max_colors': args.colors,
            'min_area': args.min_area,
            'epsilon': args.epsilon,
        })
        if res.abstained:
            print(f'Specialist abstained: {res.output.get("reason")}', file=sys.stderr)
            return 1
        Path(args.output).write_text(res.output['svg'], encoding='utf-8')
        print(f'Wrote SVG with {res.output["region_count"]} regions to {args.output}')
        return 0
        
    elif args.command == 'mogrify':
        ops = []
        if args.resize:
            ops.append({'operation': 'resize', 'geometry': args.resize, 'method': 'nano_edge'})
        if args.filter_type:
            ops.append({'operation': 'filter', 'filter_type': args.filter_type})
        if args.tone_mode:
            ops.append({'operation': 'tone', 'mode': args.tone_mode})
        if args.quantize_colors:
            ops.append({'operation': 'quantize', 'colors': args.quantize_colors, 'dither': 'nano_tone'})
        files = [Path(p) for p in args.inputs]
        out_dir = Path(args.output_dir) if args.output_dir else None
        report = mogrify_batch(files, ops, output_dir=out_dir, output_format=args.format, do_auto_orient=args.auto_orient)
        print(json.dumps(report, indent=2))
        return 0
        
    elif args.command == 'spritesheet-slice':
        sheet = load_image(args.input)
        frames = slice_spritesheet(sheet, args.frame_width, args.frame_height)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = args.format.lstrip('.')
        for idx, frame in enumerate(frames):
            save_image(frame, out_dir / f'frame_{idx:03d}.{fmt}')
        print(f'Sliced {len(frames)} frames into {out_dir}')
        return 0
        
    elif args.command == 'spritesheet-assemble':
        frames = [load_image(p) for p in args.inputs]
        sheet = assemble_spritesheet(frames, direction=args.direction, columns=args.columns)
        save_image(sheet, args.output)
        print(f'Assembled {len(frames)} frames ({args.direction}) -> {args.output}')
        return 0
        
    img = load_image(args.input)
    query = {'image': img, 'operation': args.command}
    if args.command == 'resize':
        if args.geometry:
            in_h, in_w = shape(img)
            target_w, target_h, crop_box = parse_geometry(args.geometry, in_w, in_h)
            query.update({'width': target_w, 'height': target_h, 'method': args.method})
        else:
            if not args.width or not args.height:
                raise ValueError('resize requires either --geometry or both --width and --height')
            query.update({'width': args.width, 'height': args.height, 'method': args.method})
    elif args.command == 'quantize':
        query.update({'colors': args.colors, 'dither': args.dither})
    elif args.command == 'filter':
        query.update({'filter_type': args.filter_type, 'strength': args.strength, 'radius': args.radius, 'sigma': args.sigma})
    elif args.command == 'tone':
        query.update({'mode': args.mode, 'target_luma': args.target_luma})
    elif args.command == 'threshold':
        query.update({'mode': args.mode})
    elif args.command == 'morphology':
        query.update({'morph_op': args.op, 'radius': args.radius})
    elif args.command == 'transform':
        query.update({'transform_type': args.op, 'target_width': args.target_width, 'target_height': args.target_height})
    elif args.command == 'composite':
        overlay_img = load_image(args.overlay)
        query.update({'overlay': overlay_img, 'mode': args.mode, 'opacity': args.opacity})
    elif args.command == 'channel':
        if args.action == 'extract':
            query.update({'action': 'extract', 'channel': args.channel})
        else:
            if not args.green or not args.blue:
                raise ValueError('combine action requires --green and --blue arguments')
            g_img = load_image(args.green)
            b_img = load_image(args.blue)
            query.update({'action': 'combine', 'g_image': g_img, 'b_image': b_img})
        
    res = magick_process(query)
    if res.abstained:
        print(f'Specialist abstained: {res.output.get("reason")}', file=sys.stderr)
        return 1
        
    out_image = res.output.get('image')
    if out_image is None and 'mask' in res.output:
        mask = res.output['mask']
        out_image = [[(255, 255, 255) if v else (0, 0, 0) for v in row] for row in mask]
    if out_image is not None:
        save_image(out_image, args.output)
        print(f'Successfully wrote output to {args.output}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
