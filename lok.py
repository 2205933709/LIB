import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import math
import re
import csv
import numpy as np
import plotly.express as px
from io import BytesIO
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from plotly.graph_objs import Scatter, Layout, Figure
from collections import Counter
from collections import Counter, defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder


st.write(
    """  
    <p style="color:blue; font-size:40px; font-weight:bold;">Gen-song</p>  
    """,
    unsafe_allow_html=True
)


file_groups = defaultdict(list)
# 在侧边栏中创建一个文件上传控件
uploaded_files = st.sidebar.file_uploader('请上传 CSV 文件', type=['csv'],  accept_multiple_files=True)

# 初始化一个空的DataFrame来保存所有数据
all_data = pd.DataFrame()
if uploaded_files:
    row_count = 0
    column_order = ['序号', '文件名','SN号','开始时间', '结束时间', '任务执行用时(s)', '停车次数',
                    '故障编号', '故障出现次数', '故障出现时间段', '最大横向偏差(mm)',
                    '最大航向偏差*10(°)', '平均速度（mm/s）', '平均纠偏量（°）', '最大纠偏量（°）',
                    '停止精度（后退）(mm)', '终点横向偏差（后退）(mm)','终点航向偏差（后退）(°)','终点状态信息（后退）',
                    '终点路线号（后退）','终点路线类型（后退）',
                    '停止精度（前进）(mm)', '终点横向偏差（前进）(mm)','终点航向偏差（前进）(°)', '终点状态信息（前进）',
                    '终点路线号（前进）' ,'终点路线类型（前进）']

    for uploaded_file in uploaded_files:

        with uploaded_file as file:

            df = pd.read_csv(file, skiprows=2, on_bad_lines='skip' )


            #检查上传的csv文件是否有自动任务数据
            if 'LogType' in df.columns:
                if 10 not in df['LogType'].unique():
                    st.write(f"文件 {uploaded_file.name} 中无数据or格式错误")
                    continue  # 跳过当前循环的剩余部分，处理下一个文件


        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')


        # 提取文件名中的SN号
        filename = uploaded_file.name
        sn_number = ''
        if '-' in filename:
            parts = filename.split('-')
            if len(parts) > 1:

                sn_candidate = parts[0]
                if len(sn_candidate) >= 12:
                    sn_number = sn_candidate


        filtered_df = df
        special_points = []
        start_time = None
        end_time = None
        parking_count = 0

        # 检查 MotorCommand_SteeringAngle1 列是否存在
        steering_angle_column_exists = 'MotorCommand_SteeringAngle1' in filtered_df.columns

        # 根据列是否存在来设置route_number的索引
        route_number_index = 42 if steering_angle_column_exists else 52  # 假设第53列是route_number（索引从0开始）

        # 停止精度相关数据处理
        prev_100_speeds = []
        prev_log_type_was_10 = False

        for index, row in filtered_df.iterrows():
            log_type = row['LogType']
            remain_distance = row['RemainDistance']
            route_number = row.iloc[route_number_index]
            lateral_error = row.iloc[9]
            attitude_error = row.iloc[10]
            motor_speed = row['MotorStatus_MotorSpeed1']


            if len(prev_100_speeds) < 100:
                prev_100_speeds.append(motor_speed)
            else:
                prev_100_speeds.pop(0)
                prev_100_speeds.append(motor_speed)


            if not (-150 < remain_distance < 150):
                continue

            # 前一个LogType是10，而当前不是，那么记录前一个点的信息
            if prev_log_type_was_10 and log_type != 10:
                prev_row = filtered_df.iloc[index - 1]
                prev_remain_distance = prev_row['RemainDistance']
                prev_route_number = prev_row.iloc[route_number_index]
                prev_lateral_error = prev_row.iloc[9]
                prev_attitude_error = prev_row.iloc[10]

                
                forward_or_backward = "未知"
                if any(speed > 0 for speed in prev_100_speeds):
                    forward_or_backward = "前进"
                elif any(speed < 0 for speed in prev_100_speeds):
                    forward_or_backward = "后退"

                # 构建hovertext字符串
                hover_text = f"{forward_or_backward}/"
                if prev_remain_distance > 0:
                    hover_text += "未到点"
                elif prev_remain_distance < 0:
                    hover_text += "过冲"

                special_points.append((
                    prev_row['Time'],
                    prev_remain_distance,
                    prev_route_number,
                    prev_lateral_error,
                    prev_attitude_error,
                    hover_text
                ))

            prev_log_type_was_10 = log_type == 10

        # 故障相关数据处理
        # 确保ErrorCode列存在(故障例）
        if 'ErrorCode' in df.columns:


            fault_counter = Counter()
            fault_status = {}
            fault_times = defaultdict(lambda: {'occurrences': []})  # 修改此处以存储所有的出现时间段

            # 获取DataFrame的总行数
            total_rows = len(df)

            # 遍历ErrorCode列
            for index, row in df.iterrows():
                error_code = row['ErrorCode']
                time_stamp = row['Time']

                # 如果ErrorCode为空或者为NaN，则跳过
                if pd.isnull(error_code) or not isinstance(error_code, int):
                    continue

                    # 将错误码转换为二进制形式
                binary_representation = bin(error_code)[2:]

                for i, bit in enumerate(reversed(binary_representation)):
                    fault_number = i + 1

                    # 如果当前位是1，则故障存在
                    if bit == '1':
                        if fault_number not in fault_status or not fault_status[fault_number]:
                            # 如果故障之前不存在或已经解决，则增加计数
                            fault_counter[fault_number] += 1
                            # 更新故障状态为激活
                            fault_status[fault_number] = True
                            # 记录故障首次出现的时间
                            fault_times[fault_number]['occurrences'].append(
                                {'first_seen': time_stamp, 'last_seen': time_stamp})
                        else:
                            # 更新当前出现的故障的最后时间
                            fault_times[fault_number]['occurrences'][-1]['last_seen'] = time_stamp
                    else:
                        if fault_number in fault_status and fault_status[fault_number]:
                            fault_status[fault_number] = False

            # 初始化聚合变量
            aggregated_fault_numbers = []
            total_occurrences = 0
            time_ranges_combined = []

            # 聚合故障数据
            for fault_number, count in fault_counter.items():
                occurrences = fault_times[fault_number]['occurrences']
                time_ranges = [
                    "{}--{}".format(occ['first_seen'].strftime('%H:%M:%S'), occ['last_seen'].strftime('%H:%M:%S')) for
                    occ in occurrences]
                time_ranges_combined.extend(time_ranges)

                prefix = '70' if fault_number >= 10 else '700'
                prefixed_fault_number = prefix + str(fault_number)
                aggregated_fault_numbers.append(prefixed_fault_number)
                total_occurrences += count

            # 将时间段列表转换为单个字符串
            time_range_str_combined = ", ".join(time_ranges_combined)

        # 计算最大横向偏差和最大航向偏差的逻辑
        lateral_errors = [point[3] for point in special_points]
        attitude_errors_rad = [point[4] for point in special_points]
        attitude_errors_deg = [math.degrees(error) * 10 for error in attitude_errors_rad]

        max_lateral_error_index = max(range(len(lateral_errors)), key=lambda i: abs(lateral_errors[i]))
        max_attitude_error_index = max(range(len(attitude_errors_deg)), key=lambda i: abs(attitude_errors_deg[i]))

        max_lateral_error = lateral_errors[max_lateral_error_index]
        max_attitude_error = attitude_errors_deg[max_attitude_error_index]

        # 确定开始时间和结束时间
        for index, row in filtered_df.iterrows():
            log_type = row['LogType']
            if log_type in [10, 12, 13, 14]:
                if start_time is None:
                    start_time = row['Time']  # 设置开始时间
                end_time = row['Time']  # 更新结束时间为当前时间戳
        # 如果开始时间和结束时间都提供了，则计算执行用时
        if start_time and end_time:
            execution_time = end_time - start_time
            # 将执行用时转换为秒
            execution_time_seconds = execution_time.total_seconds()

        # 根据开始时间和结束时间筛选数据
        masked_df = filtered_df[(filtered_df['Time'] >= start_time) & (filtered_df['Time'] <= end_time)]

        # 计算平均速度
        average_speed = masked_df['MotorStatus_MotorSpeed1'].mean()

        # 检查纠偏量所需的列是否存在
        if 'MotorCommand_SteeringAngle1' in masked_df.columns:
            # 计算纠偏量
            masked_df['Deviation'] = masked_df['MotorCommand_SteeringAngle1'] - masked_df['NominalCtrlOutput']

            # 计算平均纠偏量
            average_deviation = masked_df['Deviation'].mean()

            # 计算最大纠偏量（保留正负号）
            max_deviation_index = masked_df['Deviation'].abs().idxmax()
            max_deviation_value = masked_df.at[max_deviation_index, 'Deviation']
        else:
            average_deviation = 0
            max_deviation_value = 0


        # 添加路线类型判断的函数
        def get_route_type(df, point_time, angle_column='MotorStatus_SteeringAngle1', window_size=200):
            if angle_column not in df.columns:
                # 如果列不存在，直接返回'直线'
                return '直线'
            # 获取点时间的索引
            point_index = df[df['Time'] == point_time].index[0]
            # 获取前window_size行的角度数据
            angles = df.loc[point_index - window_size:point_index, angle_column]
            # 检查是否有角度大于10或小于-10
            if angles.max() > 10 or angles.min() < -10:
                return '弯道'
            else:
                return '直线'


        # 停止精度相关数据重处理
        forward_stop_accuracies = []
        backward_stop_accuracies = []
        forward_lateral_deviations = []
        forward_heading_deviations = []
        backward_lateral_deviations = []
        backward_heading_deviations = []
        forward_statuses = []
        backward_statuses = []
        forward_route_numbers = []
        backward_route_numbers = []
        forward_route_types = []
        backward_route_types = []

        # 遍历特殊点
        for point in special_points:
            route_number = str(point[2])[-5:]  # 取路线号的后五位
            status = point[5]  # 状态信息
            stop_accuracy = round(point[1], 2)  # 停止精度保留两位小数
            lateral_deviation = round(point[3], 2)  # 终点横向偏差
            heading_deviation = round(math.degrees(point[4]), 2)  # 终点航向偏差转换为度数并保留两位小数
            stop_time = point[0]
            # 判断路线类型
            route_type = get_route_type(df, stop_time)

            # 根据状态将停止精度和其他数据分类到前进或后退列表中
            if '前进' in status:
                forward_stop_accuracies.append(stop_accuracy)
                forward_lateral_deviations.append(lateral_deviation)
                forward_heading_deviations.append(heading_deviation)
                forward_statuses.append(status.replace('前进/', ''))
                forward_route_numbers.append(route_number)
                forward_route_types.append(route_type)
            elif '后退' in status:
                backward_stop_accuracies.append(stop_accuracy)
                backward_lateral_deviations.append(lateral_deviation)
                backward_heading_deviations.append(heading_deviation)
                backward_statuses.append(status.replace('后退/', ''))
                backward_route_numbers.append(route_number)
                backward_route_types.append(route_type)


        # 转换列表为字符串，用于后续的数据展示或存储
        def list_to_str(lst):
            return ' ; '.join(map(str, lst)) if lst else ''

        data = {
            '文件名': uploaded_file.name,
            'SN号': [sn_number],
            '开始时间': [start_time] if start_time else [],
            '结束时间': [end_time] if end_time else [],
            '任务执行用时(s)': round(execution_time_seconds, 2),
            '停车次数': [len(special_points)],
            '故障编号': ", ".join(aggregated_fault_numbers),
            '故障出现次数': total_occurrences,
            '故障出现时间段': time_range_str_combined,
            '最大横向偏差(mm)': [round(max_lateral_error, 2)],
            '最大航向偏差*10(°)': [round(max_attitude_error, 2)],
            '平均速度（mm/s）': [round(average_speed, 2)],
            '平均纠偏量（°）': [round(average_deviation, 2)],
            '最大纠偏量（°）': [round(max_deviation_value, 2)],
            '停止精度（后退）(mm)': list_to_str(backward_stop_accuracies),
            '终点横向偏差（后退）(mm)': list_to_str(backward_lateral_deviations),
            '终点航向偏差（后退）(°)': list_to_str(backward_heading_deviations),
            '终点状态信息（后退）': list_to_str(backward_statuses),
            '终点路线号（后退）': list_to_str(backward_route_numbers),
            '终点路线类型（后退）': '; '.join(backward_route_types),
            '停止精度（前进）(mm)': list_to_str(forward_stop_accuracies),
            '终点横向偏差（前进）(mm)': list_to_str(forward_lateral_deviations),
            '终点航向偏差（前进）(°)': list_to_str(forward_heading_deviations),
            '终点状态信息（前进）': list_to_str(forward_statuses),
            '终点路线号（前进）': list_to_str(forward_route_numbers),
            '终点路线类型（前进）': '; '.join(forward_route_types),
        }

        # 将字典转换为DataFrame
        temp_df = pd.DataFrame(data)
        temp_df = temp_df.reindex(columns=column_order)
        # 填充序号
        row_count += 1
        temp_df.at[0, '序号'] = row_count
        all_data = pd.concat([all_data, temp_df], ignore_index=True)


        def extract_key_number(filename):
            match = re.search(r'-(\d+)-[^-]+-(\d+)', filename)
            if match:
                return match.group(2)
            return -1


        all_data['key_numbers'] = all_data['文件名'].apply(extract_key_number)

        # 创建一个空DataFrame用于存储聚合后的数据
        aggregated_data = pd.DataFrame()

        # 筛选出不含'idle'的文件进行分组和聚合
        non_idle_data = all_data[~all_data['文件名'].str.contains('idle', na=False)]
        if not non_idle_data.empty:
            # 过滤掉没有匹配关键数字的行
            non_idle_data = non_idle_data.dropna(subset=['key_numbers'])



            def max_abs_with_sign(series):
                max_abs_idx = np.abs(series).idxmax()
                return series[max_abs_idx]

            def aggregate_rows(group):

                # 字符串列，使用逗号连接
                string_cols = ['文件名','SN号' ]
                string_dict = {col: ', '.join(group[col].unique()) for col in string_cols}

                # 数字列，求和
                numeric_cols = ['停车次数', '任务执行用时(s)', '故障出现次数']
                numeric_dict = {col: group[col].sum() if col in group.columns else None for col in numeric_cols}

                # 单值列（不是列表），保留平均数
                single_value_list_cols = ['平均速度（mm/s）', '平均纠偏量（°）']
                single_value_list_dict = {
                    col: round(group[col].mean(), 2) if col in group.columns and not group[col].empty else None for col in
                    single_value_list_cols}

                #单值列，保留绝对值最大数
                max_value_cols = [
                    '最大横向偏差(mm)', '最大航向偏差*10(°)', '最大纠偏量（°）'
                ]
                max_value_dict = {col: max_abs_with_sign(group[col].dropna()) for col in max_value_cols}

                #多值列，使用分号连接
                multi_value_list_cols = [
                    '停止精度（后退）(mm)', '终点横向偏差（后退）(mm)', '终点航向偏差（后退）(°)', '终点状态信息（后退）', '终点路线号（后退）',
                    '停止精度（前进）(mm)', '终点横向偏差（前进）(mm)', '终点航向偏差（前进）(°)', '终点状态信息（前进）', '终点路线号（前进）'
                    , '终点路线类型（后退）', '终点路线类型（前进）' ,'故障编号', '故障出现时间段'
                ]
                multi_value_list_dict = {
                    col: '; '.join(
                        item for sublist in group[col].dropna().tolist() for item in sublist.split(';') if item)
                    for col in multi_value_list_cols
                }

                # 开始时间和结束时间处理
                time_cols = ['开始时间', '结束时间']

                # 首先，确保时间列是datetime类型
                for col in time_cols:
                    group[col] = pd.to_datetime(group[col], format='%d/%m/%Y %H:%M',
                                                errors='coerce')  # errors='coerce'会将无法转换的值设为NaT

                # 取最早和最晚的时间
                time_dict = {
                    '开始时间': group['开始时间'].min() if not group['开始时间'].empty else None,
                    '结束时间': group['结束时间'].max() if not group['结束时间'].empty else None
                }

                # 将所有字典合并成一个
                group_dict = {**string_dict, **numeric_dict, **single_value_list_dict,
                              **multi_value_list_dict, **time_dict,**max_value_dict}

                return pd.Series(group_dict)

            # 分组并应用 aggregate_rows 函数
            aggregated_data = non_idle_data.groupby('key_numbers').apply(aggregate_rows).reset_index()

            # 删除不再需要的列
            aggregated_data = aggregated_data.drop(columns=['key_numbers'])

        # 将包含'idle'的文件添加到结果中
        idle_data = all_data[all_data['文件名'].str.contains('idle', na=False)]


        if aggregated_data.empty:
            merged_data = idle_data
        else:

            merged_data = pd.concat([aggregated_data, idle_data], ignore_index=True)

        # 合并后的数据集生成序号
        merged_data['序号'] = range(1, len(merged_data) + 1)

    # Configure grid options using GridOptionsBuilder
    builder = GridOptionsBuilder.from_dataframe(all_data)
    builder.configure_pagination(enabled=True)
    builder.configure_selection(selection_mode='single', use_checkbox=True)
    grid_options = builder.build()


    st.write("数据汇总：")
    ag_grid = AgGrid(merged_data, gridOptions=grid_options)

    if st.button('筛选数据'):
        filtered_data = []

        for index, row in all_data.iterrows():
            # 解析停止精度（后退和前进）
            backward_accuracies = [float(acc) for acc in row['停止精度（后退）(mm)'].split(';') if acc]
            forward_accuracies = [float(acc) for acc in row['停止精度（前进）(mm)'].split(';') if acc]

            # 解析终点横向偏差（后退和前进）
            backward_lateral_deviations = [float(dev) for dev in row['终点横向偏差（后退）(mm)'].split(';') if dev]
            forward_lateral_deviations = [float(dev) for dev in row['终点横向偏差（前进）(mm)'].split(';') if dev]

            # 解析终点航向偏差（后退和前进）
            backward_heading_deviations = [float(dev) for dev in row['终点航向偏差（后退）(°)'].split(';') if dev]
            forward_heading_deviations = [float(dev) for dev in row['终点航向偏差（前进）(°)'].split(';') if dev]

            # 解析终点状态信息、终点路线号、终点路线类型（后退和前进）
            backward_end_state_infos = row['终点状态信息（后退）'].split(';') if pd.notnull(row['终点状态信息（后退）']) else []
            forward_end_state_infos = row['终点状态信息（前进）'].split(';') if pd.notnull(row['终点状态信息（前进）']) else []
            backward_end_route_numbers = row['终点路线号（后退）'].split(';') if pd.notnull(row['终点路线号（后退）']) else []
            forward_end_route_numbers = row['终点路线号（前进）'].split(';') if pd.notnull(row['终点路线号（前进）']) else []
            backward_end_route_types = row['终点路线类型（后退）'].split(';') if pd.notnull(row['终点路线类型（后退）']) else []
            forward_end_route_types = row['终点路线类型（前进）'].split(';') if pd.notnull(row['终点路线类型（前进）']) else []


            # 检查是否有值超出范围，并记录相关信息
            out_of_range_info = {}
            if row['停止精度（后退）(mm)'] is not None and row['停止精度（后退）(mm)'] != '':
                if (any(acc > 20 for acc in backward_accuracies) or any(
                        acc < -20 for acc in backward_accuracies)):
                    out_of_range_info['停止精度（后退）(mm)'] = row['停止精度（后退）(mm)']
                    out_of_range_info['终点横向偏差（后退）(mm)'] = row['终点横向偏差（后退）(mm)']
                    out_of_range_info['终点航向偏差（后退）(°)'] = row['终点航向偏差（后退）(°)']
            if row['停止精度（前进）(mm)'] is not None and row['停止精度（前进）(mm)'] != '':
                if (any(acc > 20 for acc in forward_accuracies) or any(
                        acc < -20 for acc in forward_accuracies)):
                    out_of_range_info['停止精度（前进）(mm)'] = row['停止精度（前进）(mm)']
                    out_of_range_info['终点横向偏差（前进）(mm)'] = row['终点横向偏差（前进）(mm)']
                    out_of_range_info['终点航向偏差（前进）(°)'] = row['终点航向偏差（前进）(°)']
            if row['终点航向偏差（后退）(°)'] is not None and row['终点航向偏差（后退）(°)'] != '':
                if (any(dev > 2 for dev in backward_heading_deviations) or any(
                        dev < -2 for dev in backward_heading_deviations)):
                    if '停止精度（后退）(mm)' not in out_of_range_info:
                        out_of_range_info['停止精度（后退）(mm)'] = row['停止精度（后退）(mm)']
                    if '终点横向偏差（后退）(mm)' not in out_of_range_info:
                        out_of_range_info['终点横向偏差（后退）(mm)'] = row['终点横向偏差（后退）(mm)']
                    out_of_range_info['终点航向偏差（后退）(°)'] = row['终点航向偏差（后退）(°)']
            if row['终点航向偏差（前进）(°)'] is not None and row['终点航向偏差（前进）(°)'] != '':
                if (any(dev > 2 for dev in forward_heading_deviations) or any(
                        dev < -2 for dev in forward_heading_deviations)):
                    if '停止精度（前进）(mm)' not in out_of_range_info:
                        out_of_range_info['停止精度（前进）(mm)'] = row['停止精度（前进）(mm)']
                    if '终点横向偏差（前进）(mm)' not in out_of_range_info:
                        out_of_range_info['终点横向偏差（前进）(mm)'] = row['终点横向偏差（前进）(mm)']
                    out_of_range_info['终点航向偏差（前进）(°)'] = row['终点航向偏差（前进）(°)']
            if row['终点横向偏差（后退）(mm)'] is not None and row['终点横向偏差（后退）(mm)'] != '':
                if (any(dev > 20 for dev in backward_lateral_deviations) or any(
                        dev < -20 for dev in backward_lateral_deviations)):
                    if '停止精度（后退）(mm)' not in out_of_range_info:
                        out_of_range_info['停止精度（后退）(mm)'] = row['停止精度（后退）(mm)']
                    if '终点航向偏差（后退）(°)' not in out_of_range_info:
                        out_of_range_info['终点航向偏差（后退）(°)'] = row['终点航向偏差（后退）(°)']
                    out_of_range_info['终点横向偏差（后退）(mm)'] = row['终点横向偏差（后退）(mm)']
            if row['终点横向偏差（前进）(mm)'] is not None and row['终点横向偏差（前进）(mm)'] != '':
                if (any(dev > 20 for dev in forward_lateral_deviations) or any(
                        dev < -20 for dev in forward_lateral_deviations)):
                    if '停止精度（前进）(mm)' not in out_of_range_info:
                        out_of_range_info['停止精度（前进）(mm)'] = row['停止精度（前进）(mm)']
                    if '终点航向偏差（前进）(°)' not in out_of_range_info:
                        out_of_range_info['终点航向偏差（前进）(°)'] = row['终点航向偏差（前进）(°)']
                    out_of_range_info['终点横向偏差（前进）(mm)'] = row['终点横向偏差（前进）(mm)']


            if out_of_range_info:  # 如果有超出范围的数据，则添加到筛选结果中

                if '停止精度（后退）(mm)' in out_of_range_info and row['停止精度（后退）(mm)'] is not None and row[
                    '停止精度（后退）(mm)'] != '':
                    if backward_end_state_infos:
                        out_of_range_info['终点状态信息（后退）'] = ';'.join(backward_end_state_infos)
                    if backward_end_route_numbers:
                        out_of_range_info['终点路线号（后退）'] = ';'.join(backward_end_route_numbers)
                    if backward_end_route_types:
                        out_of_range_info['终点路线类型（后退）'] = ';'.join(backward_end_route_types)

                        # 前进的相关数据
                if '停止精度（前进）(mm)' in out_of_range_info and row['停止精度（前进）(mm)'] is not None and row[
                    '停止精度（前进）(mm)'] != '':
                    if forward_end_state_infos:
                        out_of_range_info['终点状态信息（前进）'] = ';'.join(forward_end_state_infos)
                    if forward_end_route_numbers:
                        out_of_range_info['终点路线号（前进）'] = ';'.join(forward_end_route_numbers)
                    if forward_end_route_types:
                        out_of_range_info['终点路线类型（前进）'] = ';'.join(forward_end_route_types)

                filtered_data.append({'序号': index, '文件名': row['文件名'], **out_of_range_info})



        all_columns = [
            '序号', '文件名',
            '停止精度（后退）(mm)', '终点横向偏差（后退）(mm)', '终点航向偏差（后退）(°)',
            '停止精度（前进）(mm)', '终点横向偏差（前进）(mm)', '终点航向偏差（前进）(°)',
            '终点状态信息（后退）', '终点路线号（后退）', '终点路线类型（后退）',
            '终点状态信息（前进）', '终点路线号（前进）', '终点路线类型（前进）'
        ]


        actual_columns = []

        if filtered_data:

            actual_columns = [col for col in all_columns if col in filtered_data[0].keys()]

        # 调整列的顺序
        adjusted_columns = []
        if actual_columns:
            for col in actual_columns:
                if col == '终点航向偏差（后退）(°)':
                    # 在“终点航向偏差（后退）（°）”之后插入相关列
                    adjusted_columns.append(col)
                    adjusted_columns.extend([
                        '终点状态信息（后退）',
                        '终点路线号（后退）',
                        '终点路线类型（后退）'
                    ])
                elif col not in [
                    '终点状态信息（后退）',
                    '终点路线号（后退）',
                    '终点路线类型（后退）'
                ]:
                    adjusted_columns.append(col)
        else:
            adjusted_columns = []


        filtered_df = pd.DataFrame(filtered_data, columns=adjusted_columns)

        # 检查筛选结果是否为空
        if filtered_df.empty:
            st.write('精度符合标准')
        else:
            # 显示筛选结果
            st.write('筛选结果：')
            st.write(filtered_df)


    # Display AgGrid
    # 列名列表，这些列将被特殊处理并绘制散点图
    scatter_plot_columns = [
        '停止精度（后退）(mm)',
        '终点横向偏差（后退）(mm)',
        '停止精度（前进）(mm)',
        '终点横向偏差（前进）(mm)',

    ]

    scatter_plot_angle_columns = [
        '终点航向偏差（后退）(°)',
        '终点航向偏差（前进）(°)'
    ]

    # 索引是数字，将其转换为“任务N”格式
    merged_data.index = ["任务" + str(i + 1) for i in merged_data.index]

    # 列出需要从选择框中排除的列名
    excluded_columns = [
        '序号', '文件名','SN号' ,'开始时间', '结束时间', '故障编号', '故障出现时间段',
        '终点状态信息（后退）', '终点路线号（后退）', '终点路线类型（后退）',
        '终点状态信息（前进）', '终点路线号（前进）', '终点路线类型（前进）'
    ]

    # 从 all_data.columns 中排除这些列，得到一个新的列名列表
    selectable_columns = [col for col in merged_data.columns if col not in excluded_columns]

    # 使用筛选后的列名列表创建选择框
    selected_column = st.sidebar.selectbox('选择列以绘制图表', selectable_columns)


    # 如果选择了一列，则根据列名绘制相应的图表

    if selected_column:
        if selected_column in scatter_plot_columns:

            interval_counts = {
                '-40以下': 0,
                '-40到-20': 0,
                '-20到0': 0,
                '0到20': 0,
                '20到40': 0,
                '40以上': 0
            }


            for index, row in merged_data.iterrows():

                values_str = str(row[selected_column]).split(';')
                for value_str in values_str:

                    value_str = value_str.strip()
                    if not value_str or not value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                        continue
                    value = float(value_str)

                    # Update interval counter
                    if value < -40:
                        interval_counts['-40以下'] += 1
                    elif -40 <= value < -20:
                        interval_counts['-40到-20'] += 1
                    elif -20 <= value < 0:
                        interval_counts['-20到0'] += 1
                    elif 0 <= value < 20:
                        interval_counts['0到20'] += 1
                    elif 20 <= value <= 40:
                        interval_counts['20到40'] += 1
                    else:
                        interval_counts['40以上'] += 1


            labels = list(interval_counts.keys())
            values = list(interval_counts.values())


            total_count = sum(values)

            # 计算-20到0和0到20的百分比
            range_20_to_0_percent = (interval_counts['-20到0'] + interval_counts[
                '0到20']) / total_count * 100 if total_count > 0 else 0

            # 评级逻辑
            if range_20_to_0_percent > 70:
                rating = '优秀'
            elif 40 <= range_20_to_0_percent <= 70:
                rating = '良好'
            elif 20 <= range_20_to_0_percent < 40:
                rating = '中等'
            else:
                rating = '较差'

            # 打印评级
            print(f"评级: {rating}")


            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(title_text=f'{selected_column} (评级: {rating})')

            # 显示饼状图
            st.plotly_chart(fig)

        elif selected_column in scatter_plot_angle_columns:

            interval_counts = {
                '-4以下': 0,
                '-4到-2': 0,
                '-2到0': 0,
                '0到2': 0,
                '2到4': 0,
                '4以上': 0
            }


            for index, row in merged_data.iterrows():

                values_str = str(row[selected_column]).split(';')
                for value_str in values_str:

                    value_str = value_str.strip()
                    if not value_str or not value_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                        continue
                    value = float(value_str)


                    if value < -4:
                        interval_counts['-4以下'] += 1
                    elif -4 <= value < -2:
                        interval_counts['-4到-2'] += 1
                    elif -2 <= value < 0:
                        interval_counts['-2到0'] += 1
                    elif 0 <= value < 2:
                        interval_counts['0到2'] += 1
                    elif 2 <= value <= 4:
                        interval_counts['2到4'] += 1
                    else:
                        interval_counts['4以上'] += 1


            labels = list(interval_counts.keys())
            values = list(interval_counts.values())


            total_count = sum(values)

            # 计算-2到0和0到2的百分比
            range_20_to_0_percent = (interval_counts['-2到0'] + interval_counts[
                '0到2']) / total_count * 100 if total_count > 0 else 0

            # 评级逻辑
            if range_20_to_0_percent > 70:
                rating = '优秀'
            elif 40 <= range_20_to_0_percent <= 70:
                rating = '良好'
            elif 20 <= range_20_to_0_percent < 40:
                rating = '中等'
            else:
                rating = '较差'


            # 打印评级
            print(f"评级: {rating}")


            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(title_text=f'{selected_column} (评级: {rating})')

            # 显示饼状图
            st.plotly_chart(fig)

        else:
            # 对于非饼状图列，绘制折线图
            fig = go.Figure(data=go.Scatter(x=merged_data.index, y=merged_data[selected_column], mode='lines'))
            fig.update_layout(title_text=f'{selected_column}')
            st.plotly_chart(fig)

    # 运行 Streamlit 应用
if __name__ == '__main__':
    st.write('🌈 Welcome! 🌈')