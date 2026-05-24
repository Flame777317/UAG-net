def analyze_comprehensive_features(self, x, mask, b, save_path='./analysis'):
    """全面的特征分析：x_last, inter_xk, 每个阶段UBlock输入输出"""
    self.model.eval()
    with torch.no_grad():
        # 手动执行完整流程来获取所有特征
        inter_xk = []
        xfb = None
        x_last = torch.zeros_like(x)
        y = x
        t = 1

        print("=== Comprehensive Feature Analysis ===")

        for i in range(self.model.layer_num):
            print(f"\n--- Stage {i + 1} ---")

            # 获取当前阶段的输入
            device = x.device
            z = self.model.denoise_stage[i].A(y, self.model.rate, mask)
            r = y - self.model.denoise_stage[i].tau * self.model.denoise_stage[i].AT(
                z - b * (z / (torch.abs(z) + 1e-8)), self.model.rate, mask)
            x_f = self.model.denoise_stage[i].conv_forward(r) + x_last

            # UBlock输入特征
            print(f"UBlock Input - Channels: {x_f.shape[1]}, L2 norm: {torch.norm(x_f[0]).item():.3f}")
            save_l2_norm_pseudocolor(x_f[0].cpu(), f'{save_path}/stages', f'stage{i + 1}_ublock_input.png')


import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def save_l2_norm_pseudocolor(tensor_data, output_folder, output_filename='l2_norm_pseudocolor.png', cmap='jet'):
    """
     将一个形状为 (C, H, W) 的 Tensor 的 L2 范数计算并保存为伪彩色图像。
     统一使用jet colormap便于特征对比
     """
    if tensor_data.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional (C, H, W)")

    # 将 Tensor 转换为 NumPy 数组
    numpy_data = tensor_data.permute(1, 2, 0).numpy()

    # 在通道维度上计算L2范数
    l2_norm = np.linalg.norm(numpy_data, axis=-1)

    # 创建保存图像的文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 增强对比度的归一化
    l2_min, l2_max = l2_norm.min(), l2_norm.max()
    if l2_max > l2_min:
        # 归一化到0-1
        l2_norm_normalized = (l2_norm - l2_min) / (l2_max - l2_min)
        # 轻微的gamma校正增强对比度
        l2_norm_normalized = np.power(l2_norm_normalized, 0.8)
    else:
        l2_norm_normalized = l2_norm

    # 统一使用jet colormap
    output_file_path = os.path.join(output_folder, output_filename)
    plt.figure(figsize=(4, 4))
    plt.imshow(l2_norm_normalized, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    # 确保没有边框和padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    print(f"Feature map (jet) saved to {output_file_path} with shape {l2_norm.shape}")


def create_professional_visualization(save_path='./analysis', original_img_path=None):
    """创建高质量的模块有效性可视化图"""
    import cv2
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available, using simple text overlay")
        return create_simple_visualization(save_path, original_img_path)

    # 图像尺寸设置
    img_size = (200, 200)
    margin = 12
    row_label_width = 40

    # 严格两行，每行5张图
    n_cols = 5
    n_rows = 2
    total_width = row_label_width + n_cols * img_size[0] + (n_cols + 1) * margin
    total_height = n_rows * img_size[1] + margin * 3

    # 创建纯白背景画布
    canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

    # 第一行：UBlock输出的演化过程
    row1_images = [
        f'{save_path}/stages/stage1_ublock_output.png',
        f'{save_path}/stages/stage4_ublock_output.png',
        f'{save_path}/stages/stage6_ublock_output.png',
        f'{save_path}/stages/stage7_ublock_output.png',
        original_img_path  # 原图
    ]

    # 第二行：特征融合过程
    row2_images = [
        f'{save_path}/fusion/x_last_for_concat.png',
        f'{save_path}/fusion/inter_xk_processed.png',
        f'{save_path}/fusion/concatenated_features.png',
        f'{save_path}/fusion/feature_merged.png',
        None  # 残差图
    ]

    all_rows = [
        ('UBlock', row1_images),
        ('Fusion', row2_images)
    ]

    for row_idx, (row_name, images) in enumerate(all_rows):
        # 计算每行图片的实际起始位置
        y_start = margin + row_idx * (img_size[1] + margin)

        # 添加竖向行标签 - 精确居中对齐
        text_center_y = y_start + img_size[1] // 2
        text_start_y = text_center_y - (len(row_name) * 10)  # 根据字符数调整起始位置

        canvas = add_vertical_text(canvas, row_name, (20, text_start_y),
                                   font_size=18, color=(0, 0, 0))

        # 处理每张图片
        for col_idx, img_path in enumerate(images):
            x_start = row_label_width + col_idx * (img_size[0] + margin)
            y_end = y_start + img_size[1]

            if img_path is None:  # 残差图
                if original_img_path and os.path.exists(original_img_path):
                    residual_img = generate_error_map(original_img_path, f'{save_path}/output/reconstructed_image.png')
                    if residual_img is not None:
                        img_resized = cv2.resize(residual_img, img_size)
                        canvas[y_start:y_end, x_start:x_start + img_size[0]] = img_resized
            elif img_path and os.path.exists(img_path):
                # 读取图片
                if img_path == original_img_path:
                    # 原图处理
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif 'reconstructed_image' in img_path:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.imread(img_path)

                if img is not None:
                    # 使用INTER_AREA避免产生边框，保持图像质量
                    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                    canvas[y_start:y_end, x_start:x_start + img_size[0]] = img_resized

    # 保存高质量结果
    os.makedirs(f'{save_path}/visualization', exist_ok=True)

    try:
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        canvas_pil.save(f'{save_path}/visualization/professional_analysis.png',
                        'PNG', dpi=(300, 300), optimize=True)
        print(f"High-quality visualization (300 DPI) saved to {save_path}/visualization/professional_analysis.png")
    except:
        cv2.imwrite(f'{save_path}/visualization/professional_analysis.png', canvas)
        print(f"Visualization saved to {save_path}/visualization/professional_analysis.png")

    return canvas


def add_vertical_text(img, text, position, font_size=18, color=(0, 0, 0)):
    """添加更清晰的竖向文本"""
    import cv2
    try:
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            # 尝试使用更大的字体
            font = ImageFont.truetype("times.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

        # 竖向绘制文字，字间距适当增加
        y_offset = 0
        for char in text:
            draw.text((position[0], position[1] + y_offset), char, font=font, fill=color)
            y_offset += font_size + 3  # 增加字间距

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        # 备选方案：使用OpenCV，增大字体
        y_offset = 0
        for char in text:
            cv2.putText(img, char, (position[0], position[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  # 增大字体和粗细
            y_offset += 25
        return img


def generate_error_map(original_path, reconstructed_path):
    """生成残差图"""
    import cv2
    if not (os.path.exists(original_path) and os.path.exists(reconstructed_path)):
        return None

    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    reconstructed = cv2.imread(reconstructed_path, cv2.IMREAD_GRAYSCALE)

    if original.shape != reconstructed.shape:
        return None

    # 计算残差
    residual = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    residual = (residual / residual.max() * 255).astype(np.uint8)
    residual_colored = cv2.applyColorMap(residual, cv2.COLORMAP_JET)

    return residual_colored


def create_simple_visualization(save_path='./analysis', original_img_path=None):
    """简化版可视化（不依赖PIL）"""
    import cv2

    # 只拼接图片，不添加复杂文字
    img_size = (128, 128)
    margin = 5

    # 收集所有图片
    all_images = []

    # 第一行：UBlock分析
    ublock_imgs = []
    ublock_paths = [
        f'{save_path}/stages/stage2_ublock_input.png',
        f'{save_path}/stages/stage2_ublock_output.png',
        f'{save_path}/stages/stage3_ublock_input.png',
        f'{save_path}/stages/stage3_ublock_output.png',
        f'{save_path}/output/reconstructed_image.png'
    ]

    for path in ublock_paths:
        if os.path.exists(path):
            if 'reconstructed_image' in path:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.imread(path)
            if img is not None:
                ublock_imgs.append(cv2.resize(img, img_size))

    # 第二行：融合分析
    fusion_imgs = []
    fusion_paths = [
        f'{save_path}/fusion/x_last_for_concat.png',
        f'{save_path}/fusion/inter_xk_processed.png',
        f'{save_path}/fusion/concatenated_features.png',
        f'{save_path}/fusion/feature_merged.png',
        f'{save_path}/output/reconstructed_image.png'
    ]

    for path in fusion_paths:
        if os.path.exists(path):
            if 'reconstructed_image' in path:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.imread(path)
            if img is not None:
                fusion_imgs.append(cv2.resize(img, img_size))

    # 拼接图片
    result_rows = []
    if ublock_imgs:
        row1 = np.hstack(
            [np.concatenate([img, np.ones((img_size[1], margin, 3), dtype=np.uint8) * 255], axis=1) for img in
             ublock_imgs[:-1]] + [ublock_imgs[-1]])
        result_rows.append(row1)

    if fusion_imgs:
        row2 = np.hstack(
            [np.concatenate([img, np.ones((img_size[1], margin, 3), dtype=np.uint8) * 255], axis=1) for img in
             fusion_imgs[:-1]] + [fusion_imgs[-1]])
        result_rows.append(row2)

    if result_rows:
        final_result = np.vstack(
            [np.concatenate([row, np.ones((margin, row.shape[1], 3), dtype=np.uint8) * 255], axis=0) for row in
             result_rows[:-1]] + [result_rows[-1]])

        os.makedirs(f'{save_path}/visualization', exist_ok=True)
        cv2.imwrite(f'{save_path}/visualization/simple_analysis.png', final_result)
        print(f"Simple visualization saved to {save_path}/visualization/simple_analysis.png")
        return final_result

    return None


def add_text_to_image(img, text, position, font_size=12, color=(0, 0, 0), center=False):
    """在图像上添加文本"""
    import cv2
    try:
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            # 尝试使用Times New Roman字体
            font = ImageFont.truetype("times.ttf", font_size)
        except:
            try:
                # 尝试Linux系统的Times字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", font_size)
            except:
                try:
                    # 尝试其他常见字体
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", font_size)
                except:
                    # 使用默认字体
                    font = ImageFont.load_default()

        if center:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            position = (position[0] - text_width // 2, position[1])

        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        # 如果PIL不可用，使用OpenCV添加文字
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return img


class FeatureAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.features = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # 如果输出是tuple，只保存第一个元素
                self.features[name] = output[0].detach()
            else:
                self.features[name] = output.detach()

        return hook

    def register_hooks(self):
        """注册关键特征提取点的hooks"""
        # 特征融合有效性验证点
        self.model.mid.register_forward_hook(self.hook_fn('mid_fusion'))
        self.model.feature_merge.register_forward_hook(self.hook_fn('feature_merge'))
        self.model.post.register_forward_hook(self.hook_fn('post_process'))

    def analyze_comprehensive_features(self, x, mask, b, save_path='./analysis'):
        """全面的特征分析：展示所有UBlock输出和特征融合"""
        self.model.eval()
        with torch.no_grad():
            # 手动执行完整流程来获取所有特征
            inter_xk = []
            xfb = None
            x_last = torch.zeros_like(x)
            y = x
            t = 1

            print("=== Comprehensive Feature Analysis ===")

            # 执行所有迭代阶段，保存所有UBlock输出
            for i in range(self.model.layer_num):
                print(f"\n--- Stage {i + 1} ---")

                # 获取当前阶段的输入
                device = x.device
                z = self.model.denoise_stage[i].A(y, self.model.rate, mask)
                r = y - self.model.denoise_stage[i].tau * self.model.denoise_stage[i].AT(
                    z - b * (z / (torch.abs(z) + 1e-8)), self.model.rate, mask)

                x_f = self.model.denoise_stage[i].conv_forward(r) + x_last

                # 经过UBlock
                x_m, xfb = self.model.denoise_stage[i].unet_layer(x_f, xfb)

                # 保存UBlock输出特征 - 统一使用jet colormap
                print(f"UBlock Output - Channels: {x_m.shape[1]}, L2 norm: {torch.norm(x_m[0]).item():.3f}")
                save_l2_norm_pseudocolor(x_m[0].cpu(), f'{save_path}/stages', f'stage{i + 1}_ublock_output.png')

                # 完成当前阶段
                x_b = self.model.denoise_stage[i].conv_backward(x_m) + r
                t_next = (1 + math.sqrt(1 + 4 * t * t)) / 2
                y_next = x_b + ((t - 1) / t_next) * (x_b - x) * self.model.denoise_stage[i].fis

                # 保存当前阶段结果
                inter_xk.append(x_b)
                x_last = x_m  # 更新x_last
                x, y, t = x_b, y_next, t_next

            # 特征融合阶段分析 - 统一使用jet colormap
            print(f"\n=== Feature Fusion Analysis ===")

            # inter_xk特征 (拼接前)
            inter_xk_concat = torch.cat(inter_xk, dim=1)
            print(
                f"inter_xk concat - Channels: {inter_xk_concat.shape[1]}, L2 norm: {torch.norm(inter_xk_concat[0]).item():.3f}")
            save_l2_norm_pseudocolor(inter_xk_concat[0].cpu(), f'{save_path}/fusion', 'inter_xk_concat.png')

            # 经过mid处理的inter_xk
            inter_xk_processed = self.model.mid(inter_xk_concat)
            print(
                f"inter_xk processed - Channels: {inter_xk_processed.shape[1]}, L2 norm: {torch.norm(inter_xk_processed[0]).item():.3f}")
            save_l2_norm_pseudocolor(inter_xk_processed[0].cpu(), f'{save_path}/fusion', 'inter_xk_processed.png')

            # x_last特征 (用于拼接的)
            print(f"x_last (for concat) - Channels: {x_last.shape[1]}, L2 norm: {torch.norm(x_last[0]).item():.3f}")
            save_l2_norm_pseudocolor(x_last[0].cpu(), f'{save_path}/fusion', 'x_last_for_concat.png')

            # 拼接后的特征 xk = torch.cat([x_last, inter_xk], dim=1)
            xk = torch.cat([x_last, inter_xk_processed], dim=1)
            print(f"Concatenated features - Channels: {xk.shape[1]}, L2 norm: {torch.norm(xk[0]).item():.3f}")
            save_l2_norm_pseudocolor(xk[0].cpu(), f'{save_path}/fusion', 'concatenated_features.png')

            # 经过feature_merge
            xk_merged, _ = self.model.feature_merge(xk, None)
            print(f"Feature merged - Channels: {xk_merged.shape[1]}, L2 norm: {torch.norm(xk_merged[0]).item():.3f}")
            save_l2_norm_pseudocolor(xk_merged[0].cpu(), f'{save_path}/fusion', 'feature_merged.png')

            # 最终输出 - 保存为原图格式
            final_output = self.model.post(xk_merged)
            print(f"Final output - Shape: {final_output.shape}")

            # 保存最终重建图像（而不是特征图）
            output_img = final_output[0, 0].cpu().numpy()  # 取第一个batch的第一个通道
            output_img = np.clip(output_img, 0, 1) * 255  # 转换到0-255范围
            output_img = output_img.astype(np.uint8)  # 转换为uint8格式

            os.makedirs(f'{save_path}/output', exist_ok=True)

            # 使用cv2保存，保持原始尺寸128x128
            import cv2
            cv2.imwrite(f'{save_path}/output/reconstructed_image.png', output_img)

            print(f"Reconstructed image saved to {save_path}/output/reconstructed_image.png")
            print(f"Output image size: {output_img.shape}")

            return final_output

    def analyze_fusion_effectiveness(self, x, mask, b, save_path='./analysis'):
        """验证特征融合的有效性"""
        self.model.eval()
        with torch.no_grad():
            # 完整前向传播
            output = self.model(x, mask, b)

            # 获取关键融合阶段的特征
            mid_feat = self.features.get('mid_fusion', None)
            merge_feat = self.features.get('feature_merge', None)
            post_feat = self.features.get('post_process', None)

            if mid_feat is not None:
                print(
                    f"Mid fusion - Feature channels: {mid_feat.shape[1]}, L2 norm: {torch.norm(mid_feat[0]).item():.3f}")
                save_l2_norm_pseudocolor(mid_feat[0].cpu(), f'{save_path}/fusion', 'mid_fusion_features.png')

            if merge_feat is not None:
                print(
                    f"Feature merge - Feature channels: {merge_feat[0].shape[1]}, L2 norm: {torch.norm(merge_feat[0]).item():.3f}")
                save_l2_norm_pseudocolor(merge_feat[0].cpu(), f'{save_path}/fusion', 'feature_merge_features.png')

            if post_feat is not None:
                print(f"Post process - Output quality: PSNR improvement estimation")
                save_l2_norm_pseudocolor(post_feat[0].cpu(), f'{save_path}/fusion', 'final_output.png')

            # 计算融合效果
            if mid_feat is not None and merge_feat is not None:
                fusion_enhancement = torch.norm(merge_feat[0]) / (torch.norm(mid_feat[0]) + 1e-8)
                print(f"Fusion Enhancement Ratio: {fusion_enhancement:.3f}")

            return output


def quick_analysis(model_path, test_image_path, mask_path, noise_level=5):
    """快速分析函数"""
    import cv2
    import pickle
    import models
    from models.networks.mynet import A_CDP, Poisson_noise_torch

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.AUV_Net(layer_num=7, rate=4).eval().to(device)

    if os.path.exists(model_path):
        trained_model = torch.load(model_path, map_location=device)
        model.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载mask
    if os.path.exists(mask_path):
        mask = pickle.load(open(mask_path, 'rb')).to(device)
    else:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # 加载测试图像
    img = cv2.imread(test_image_path, 0) / 255.0
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    # 模拟测量数据
    b = Poisson_noise_torch(A_CDP(x, SamplingRate=4, mask=mask), alpha=noise_level)

    # 创建分析器
    analyzer = FeatureAnalyzer(model)
    analyzer.register_hooks()

    print("=== Comprehensive Feature Analysis ===")
    output = analyzer.analyze_comprehensive_features(x, mask, b)

    # 创建专业可视化图
    print("\n=== Creating Professional Visualization ===")
    create_professional_visualization(original_img_path=test_image_path)

    return output


if __name__ == "__main__":
    model_path = "/home/ghp/PycharmProjects/LPA-net/PR-YL/trained_models/4/step_num_state_dict.pth"
    mask_path = "/home/ghp/PycharmProjects/LPA-net/PR-YL/sampling_matrix/mask_4_128_test.p"
    image_path = "/home/ghp/PycharmProjects/LPA-net/dataset/PRTEST128nt/barbara.png"
    quick_analysis(model_path, image_path, mask_path)