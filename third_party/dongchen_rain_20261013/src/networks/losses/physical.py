import torch
import torch.nn as nn


def spatial_smoothness_loss(predictions):
    """空间平滑性约束 - 相邻像素的降雨量应该相近"""
    # 计算梯度
    grad_x = torch.diff(predictions, dim=-1)  # 水平梯度
    grad_y = torch.diff(predictions, dim=-2)  # 垂直梯度

    # 总变分损失 (Total Variation Loss)
    tv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

    return tv_loss


def mass_conservation_loss(rainfall_pred, radar_reflectivity):
    """质量守恒约束 - 降雨量应该与雷达反射率相关"""
    # Z-R关系: Z = a * R^b (Z是反射率，R是降雨率)
    # 一般取 a=200, b=1.6
    expected_rainfall = torch.pow(radar_reflectivity / 200.0, 1 / 1.6)
    expected_rainfall = torch.clamp(expected_rainfall, 0, 100)  # 限制范围

    # 计算相关性损失
    correlation_loss = (
        1
        - torch.corrcoef(torch.stack([rainfall_pred.flatten(), expected_rainfall.flatten()]))[0, 1]
    )

    return correlation_loss


def topographic_constraint_loss(rainfall_pred, elevation_data, coords):
    """地形约束 - 考虑海拔对降雨的影响"""
    # 迎风坡效应：海拔越高，降雨一般越多
    elevation_normalized = (elevation_data - elevation_data.min()) / (
        elevation_data.max() - elevation_data.min()
    )

    # 期望的地形降雨关系（简化模型）
    expected_topo_effect = 1 + 0.5 * elevation_normalized  # 海拔每增加1000m，降雨增加50%

    # 计算实际降雨与地形预期的差异
    topo_loss = torch.mean(torch.abs(rainfall_pred - rainfall_pred * expected_topo_effect))

    return topo_loss


def compute_wind_direction_consistency(rainfall, wind_u, wind_v):
    """风向一致性约束"""
    # 计算降雨梯度
    grad_x = torch.diff(rainfall, dim=-1)
    grad_y = torch.diff(rainfall, dim=-2)

    # 计算风向
    wind_direction = torch.atan2(wind_v, wind_u)

    # 降雨梯度应该与风向一致（简化）
    consistency_loss = torch.mean(
        torch.abs(
            torch.cos(wind_direction[..., :-1, :-1]) * grad_x
            + torch.sin(wind_direction[..., :-1, :-1]) * grad_y
        )
    )

    return consistency_loss


def meteorological_constraint_loss(rainfall_pred, satellite_temp, wind_data=None):
    """气象学约束"""
    losses = []

    # 1. 云顶温度约束：温度越低的云，降雨潜力越大
    cold_cloud_mask = satellite_temp < -40  # 冷云(°C)
    warm_cloud_mask = satellite_temp > 0  # 暖云

    # 冷云区域应该有更多降雨
    cold_cloud_loss = torch.mean(
        torch.relu(5.0 - rainfall_pred[cold_cloud_mask])  # 冷云区域降雨应该>5mm
    )

    # 暖云区域降雨应该较少
    warm_cloud_loss = torch.mean(
        torch.relu(rainfall_pred[warm_cloud_mask] - 20.0)  # 暖云区域降雨应该<20mm
    )

    losses.extend([cold_cloud_loss, warm_cloud_loss])

    # 2. 风场约束：降雨应该沿风向有一定的连续性
    # if wind_data is not None:
    #     wind_direction_loss = compute_wind_direction_consistency(rainfall_pred, wind_data)
    #     losses.append(wind_direction_loss)

    return sum(losses)


def temporal_consistency_loss(rainfall_current, rainfall_previous, threshold=10.0):
    """
    时间连续性约束 - 相邻时间步的降雨应该有一定连续性

    Args:
        rainfall_current: 当前时刻的降雨预测 (B, H, W)
        rainfall_previous: 前一时刻的降雨预测 (B, H, W)
        threshold: 允许的最大变化阈值 (mm/hour)

    Returns:
        temporal_loss: 时间连续性损失
    """
    # 计算相邻时间步的降雨变化
    temporal_diff = torch.abs(rainfall_current - rainfall_previous)

    # 超过阈值的变化给予惩罚
    excessive_change = torch.relu(temporal_diff - threshold)

    # 计算平均损失
    temporal_loss = torch.mean(excessive_change)

    return temporal_loss


def temporal_consistency_loss_advanced(
    rainfall_current, rainfall_previous, time_interval_hours=1.0, max_change_rate=15.0
):
    """
    高级时间连续性约束 - 考虑时间间隔和变化率

    Args:
        rainfall_current: 当前时刻的降雨预测 (B, H, W)
        rainfall_previous: 前一时刻的降雨预测 (B, H, W)
        time_interval_hours: 时间间隔（小时）
        max_change_rate: 最大允许变化率 (mm/hour²)

    Returns:
        temporal_loss: 时间连续性损失
    """
    # 计算变化率
    change_rate = torch.abs(rainfall_current - rainfall_previous) / time_interval_hours

    # 1. 基础变化率约束
    basic_loss = torch.mean(torch.relu(change_rate - max_change_rate))

    # 2. 空间相关的时间约束（相邻区域的变化应该相似）
    # 计算当前和前一时刻的空间梯度
    current_grad_x = torch.diff(rainfall_current, dim=-1)
    current_grad_y = torch.diff(rainfall_current, dim=-2)
    previous_grad_x = torch.diff(rainfall_previous, dim=-1)
    previous_grad_y = torch.diff(rainfall_previous, dim=-2)

    # 梯度变化损失
    grad_change_x = torch.abs(current_grad_x - previous_grad_x)
    grad_change_y = torch.abs(current_grad_y - previous_grad_y)
    spatial_temporal_loss = torch.mean(grad_change_x) + torch.mean(grad_change_y)

    # 3. 降雨强度相关约束（强降雨区域允许更大变化）
    rainfall_intensity = torch.maximum(rainfall_current, rainfall_previous)
    adaptive_threshold = max_change_rate * (1 + 0.1 * rainfall_intensity)  # 强度越大，阈值越高
    adaptive_loss = torch.mean(torch.relu(change_rate - adaptive_threshold))

    # 组合损失
    total_temporal_loss = basic_loss + 0.1 * spatial_temporal_loss + 0.5 * adaptive_loss

    return total_temporal_loss


def temporal_consistency_loss_with_physics(
    rainfall_current, rainfall_previous, radar_current, radar_previous
):
    """
    基于物理的时间连续性约束 - 结合雷达观测

    Args:
        rainfall_current/previous: 当前/前一时刻的降雨预测
        radar_current/previous: 当前/前一时刻的雷达反射率
    """
    # 1. 基础时间连续性
    basic_temporal_loss = temporal_consistency_loss(rainfall_current, rainfall_previous)

    # 2. 雷达-降雨一致性时间约束
    # 雷达的变化应该与降雨变化相关
    rainfall_change = rainfall_current - rainfall_previous
    radar_change = radar_current - radar_previous

    # 计算变化方向的一致性
    consistency = torch.sign(rainfall_change) * torch.sign(radar_change)
    inconsistency_loss = torch.mean(torch.relu(-consistency))  # 惩罚不一致的变化

    # 3. 变化幅度的合理性
    # 雷达变化大的地方，降雨变化也应该大
    radar_change_norm = torch.abs(radar_change) / (torch.std(radar_change) + 1e-8)
    rainfall_change_norm = torch.abs(rainfall_change) / (torch.std(rainfall_change) + 1e-8)
    magnitude_consistency_loss = torch.mean(torch.abs(radar_change_norm - rainfall_change_norm))

    total_loss = basic_temporal_loss + 0.2 * inconsistency_loss + 0.1 * magnitude_consistency_loss

    return total_loss


class PhysicsConstraintLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "spatial": 0.1,
            "mass_conservation": 0.2,
            "topographic": 0.05,
            "meteorological": 0.15,
            "temporal": 0.1,
        }

    def forward(self, predictions, auxiliary_data, previous_predictions=None):
        """
        Args:
            predictions: 预测的降雨量 (B, H, W)
            auxiliary_data: 辅助数据字典，包含：
                - radar: 雷达反射率
                - satellite_temp: 卫星云顶温度
                - elevation: 地形海拔
                - wind_u, wind_v: 风场数据
            previous_predictions: 前一时刻的预测（用于时间约束）
        """
        total_loss = 0

        # 1. 空间平滑性约束
        if "spatial" in self.weights:
            spatial_loss = spatial_smoothness_loss(predictions)
            total_loss += self.weights["spatial"] * spatial_loss

        # 2. 质量守恒约束
        if "mass_conservation" in self.weights and "radar" in auxiliary_data:
            mass_loss = mass_conservation_loss(predictions, auxiliary_data["radar"])
            total_loss += self.weights["mass_conservation"] * mass_loss

        # 3. 地形约束
        if "topographic" in self.weights and "elevation" in auxiliary_data:
            topo_loss = topographic_constraint_loss(
                predictions, auxiliary_data["elevation"], auxiliary_data.get("coords")
            )
            total_loss += self.weights["topographic"] * topo_loss

        # 4. 气象学约束
        if "meteorological" in self.weights:
            meteor_loss = meteorological_constraint_loss(
                predictions, auxiliary_data.get("satellite_temp"), auxiliary_data.get("wind_data")
            )
            total_loss += self.weights["meteorological"] * meteor_loss

        # 5. 时间连续性约束
        if "temporal" in self.weights and previous_predictions is not None:
            # 选择合适的时间约束函数
            if auxiliary_data.get("radar_previous") is not None:
                # 如果有前一时刻的雷达数据，使用物理约束
                temporal_loss = temporal_consistency_loss_with_physics(
                    predictions,
                    previous_predictions,
                    auxiliary_data["radar"],
                    auxiliary_data["radar_previous"],
                )
            else:
                # 否则使用高级时间约束
                temporal_loss = temporal_consistency_loss_advanced(
                    predictions,
                    previous_predictions,
                    time_interval_hours=auxiliary_data.get("time_interval", 1.0),
                )

            total_loss += self.weights["temporal"] * temporal_loss

        return total_loss
