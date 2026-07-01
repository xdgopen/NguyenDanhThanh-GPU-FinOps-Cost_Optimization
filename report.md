# Báo cáo bài lab GPU FinOps & Cost Optimization

## Thông tin sinh viên

- Họ và tên: Nguyễn Danh Thành
- MSSV: 2A202600581
- Notebook thực hiện: `notebook/gpu_finops_lab.ipynb`
- Gateway sử dụng: GPU FinOps Lab Gateway thông qua tunnel

## 1. Giới thiệu

Bài lab GPU FinOps & Cost Optimization mô phỏng một nền tảng quản lý chi phí GPU gồm nhiều dịch vụ chạy bằng Docker Compose: GPU Node Manager, Billing API, Spot Manager, Autoscaler, Cost Tracker và API Gateway. Notebook kết nối tới gateway để thực hiện các thao tác theo vòng đời FinOps: quan sát tài nguyên, submit workload, ghi nhận chi phí, dùng spot instance, autoscaling, phân tích lãng phí, tạo visualization và kiểm thử workload GPU thực tế trên Kaggle/Colab.

Mục tiêu chính của bài lab là hiểu cách GPU FinOps giúp kiểm soát chi phí hạ tầng AI/ML. Thay vì chỉ quan tâm model chạy được hay không, FinOps yêu cầu theo dõi thêm utilization, memory, power draw, thời gian chạy, loại GPU, hình thức mua tài nguyên và mức độ lãng phí. Từ đó có thể đưa ra quyết định tối ưu như right-sizing, autoscaling, dùng spot instance cho workload chịu lỗi được, dùng mixed precision để rút ngắn thời gian train và lập kế hoạch ngân sách cho dự án.

## 2. Phân tích từng phần

### Part 1-7: Mock cluster FinOps workflow

Ở phần monitoring ban đầu, cluster có 5 node với 10 GPU gồm T4, A100 và V100. Trạng thái ban đầu cho thấy 5/10 GPU đang bận, 5 GPU còn idle, average utilization đạt 41.5%, tổng memory đã dùng là 181.6 GB trên 320.0 GB và tổng power draw khoảng 848 W. Kết quả này cho thấy cluster chưa bị quá tải nhưng vẫn có một lượng tài nguyên nhàn rỗi đáng kể, đặc biệt các GPU T4 ở node-03 và node-04.

Sau khi submit các workload `train-resnet-001`, `train-bert-002`, `inference-api-003` và `train-llm-004`, toàn bộ 10/10 GPU được sử dụng và utilization tăng lên 73.5%. Đây là tín hiệu workload placement hoạt động đúng: hệ thống ưu tiên GPU theo yêu cầu, sau đó fallback sang GPU còn trống nếu cần. Việc utilization tăng từ 41.5% lên 73.5% cũng cho thấy chi phí idle giảm khi cluster được khai thác hiệu quả hơn.

Phần billing ghi nhận cả on-demand và spot workloads. Tổng chi phí sau các billing event là 2.5589 USD, tổng savings là 2.7078 USD và budget utilization chỉ 2.6% trên ngân sách 100 USD, trạng thái alert là OK. Trong các workload, `train-llm-004` dùng spot instance và tiết kiệm 1.2845 USD, thể hiện rõ lợi ích của spot cho workload training có thể checkpoint hoặc chịu preemption.

Ở phần spot instance, giá spot thay đổi theo từng loại GPU: T4 khoảng 0.2429 USD/giờ, A100 khoảng 2.7469 USD/giờ và V100 khoảng 1.4783 USD/giờ tại thời điểm chạy. Các request spot cho T4 và A100 đều được granted. Khi mô phỏng preemption, không có instance nào bị preempt trong lần chạy này, 6 instance vẫn active. Spot savings report ghi nhận spot cost 0.0607 USD, on-demand equivalent 0.2023 USD, tiết kiệm 0.1416 USD tương đương 70.0%. Điều này cho thấy spot instance đem lại savings lớn nhưng vẫn cần thiết kế workload có checkpoint, retry hoặc fallback sang on-demand.

Phần autoscaling sử dụng policy cost-aware với `scale_up_threshold` 70.0%, `scale_down_threshold` 25.0%, cooldown 30 giây, min node 2 và max node 10. Khi utilization đạt 73.5%, autoscaler quyết định scale up vì vượt ngưỡng 70.0%, số node tăng từ 5 lên 6. Sau scale up, utilization giảm còn 61.2% và 5 chu kỳ evaluation tiếp theo đều `no_action`. Đây là hành vi hợp lý: autoscaling giúp giảm áp lực lên cluster, sau đó giữ ổn định khi utilization nằm trong khoảng an toàn.

Phần cost analysis cho thấy các cost snapshot có total cost mỗi interval là 0.041944 USD, idle cost 0.001944 USD và waste khoảng 4.6%. Waste report tính trên cửa sổ gần nhất cho average waste 12.1%, total idle cost 0.046996 USD, total cost 0.401944 USD, potential monthly savings khoảng 1218.14 USD và severity LOW. Hệ thống đưa ra hai recommendation chính: dùng spot instance cho workload chịu lỗi được với estimated savings 65.0%, và scheduling các job không gấp vào off-peak để tiết kiệm khoảng 20.0%.

Dashboard tổng hợp sau autoscale ghi nhận 12 GPU trên 6 node, utilization 61.2%, 10 GPU busy và 2 GPU idle. Billing là 2.5589/100 USD, savings 2.7078 USD, spot savings 0.8513 USD và waste 12.1%. Ở workflow cuối, khi submit thêm workload nặng, utilization tăng lên 74.3%, autoscaler tiếp tục đề xuất scale up, cost interval là 0.043889 USD và waste 4.4%. Kết quả cuối workflow là total spend 2.7280 USD, total saved 2.8302 USD và budget used 2.7%. Điều này chứng minh workflow FinOps hoàn chỉnh có thể phát hiện tải tăng, autoscale, phân tích waste và áp dụng chiến lược spot để giảm chi phí.

### Part 8: Real GPU training trên Kaggle/Colab

Phần workload thực tế được chạy trên GPU Tesla T4 với 15.6 GB memory, CUDA 12.8 và pricing giả lập 0.35 USD/giờ. Diagnostic bằng `pynvml` hoạt động tốt, ban đầu GPU util là 0%, memory used khoảng 472 MB, power 10.2 W và temperature 54 C. Điều này xác nhận môi trường GPU thật đã sẵn sàng trước khi training.

Thí nghiệm dùng CIFAR-10 với 50,000 ảnh train và 10,000 ảnh test, model ResNet-18, chạy 3 epoch cho mỗi chế độ. FP32 baseline mất tổng cộng 119.0 giây, peak memory 0.82 GB, average GPU utilization 94.2%, average power 66.8 W, temperature trung bình 62.4 C và estimated cost 0.011568 USD. Accuracy tăng từ 30.2% ở epoch 1 lên 56.1% ở epoch 3.

Mixed Precision AMP mất 69.3 giây, peak memory 0.60 GB, average GPU utilization 69.7%, average power 65.9 W, temperature trung bình 76.8 C và estimated cost 0.006740 USD. Accuracy cuối đạt 56.2%, gần như tương đương FP32. So với FP32, AMP nhanh hơn 1.72 lần, tiết kiệm 0.22 GB memory và giảm chi phí 0.004828 USD, tương đương 41.7%. Khi extrapolate ở quy mô lớn, 1 ngày training có thể tiết kiệm 3.51 USD, 1 tuần tiết kiệm 24.54 USD và 1 tháng tiết kiệm 105.18 USD trên cùng mức giá T4.

Kết quả này cho thấy mixed precision là một chiến lược tối ưu rất hiệu quả vì cải thiện cả thời gian chạy và chi phí mà không làm giảm chất lượng model trong thí nghiệm này. AMP đặc biệt phù hợp cho GPU có Tensor Core như Tesla T4, A100 hoặc V100. Ngoài ra, việc gửi chi phí real GPU về gateway giúp hợp nhất dữ liệu mock cluster và workload thực tế trong cùng dashboard FinOps. Project `real-gpu-lab` ghi nhận total cost 0.013600 USD, total savings 0.004700 USD với 2 workload được report.

### Part 8.5: Advanced GPU cost optimization

Các cell 27-31 đã được chạy trong notebook nhưng output hiện tại vẫn ở trạng thái TODO, chưa có phân tích định lượng hoàn chỉnh. Cụ thể, notebook yêu cầu triển khai các hàm `analyze_multi_gpu_cost`, `forecast_project_cost`, `analyze_optimization_opportunities` và `create_advanced_finops_dashboard`, nhưng kết quả hiện chỉ hiển thị mô tả expected output thay vì bảng analysis, forecast, prioritization hoặc dashboard.

Về mặt phân tích, phần 8.5 nên được hoàn thiện theo các hướng sau:

- Multi-GPU scaling efficiency: so sánh cấu hình 1, 2, 4 và 8 GPU theo training time, total cost, speedup, scaling efficiency và cost per unit performance. Cấu hình tối ưu không nhất thiết là nhiều GPU nhất, vì communication overhead có thể làm chi phí tăng nhanh hơn hiệu năng.
- Project cost forecasting: chia dự án thành các phase như prototyping, training, tuning và evaluation; sau đó tính best case, expected case, worst case cùng confidence interval để dự phòng ngân sách.
- Optimization prioritization: xếp hạng các chiến lược như AMP, spot instance, right-sizing, autoscaling, scheduling và checkpointing theo savings, effort, risk và implementation order.
- Integrated dashboard: tổng hợp multi-GPU cost curve, scaling efficiency, forecast confidence band, phase cost breakdown, priority matrix và cumulative savings roadmap.
- Challenge strategy: với bài toán fine-tune LLM baseline 8x A100 trong 200 giờ, baseline cost là 8 * 200 * 3.67 = 5872 USD, vượt budget 5000 USD. Do đó cần kết hợp AMP, spot cho workload checkpoint được, tối ưu số GPU và scheduling để đưa chi phí về dưới ngân sách.

Do chưa có file `multi_gpu_scaling.png`, `project_forecast.png`, `optimization_roadmap.png`, `advanced_finops_dashboard.png` và chưa có screenshot Part 8.5 trong thư mục nộp bài, phần này cần được bổ sung nếu muốn đạt tiêu chí hoàn thành đầy đủ 8.5 parts.

## 3. Kết luận và học hỏi

Qua bài lab, em học được cách tiếp cận GPU FinOps từ cả góc nhìn kỹ thuật và chi phí. Các kỹ năng chính gồm theo dõi trạng thái GPU cluster, đọc utilization/memory/power/temperature, submit workload, ghi nhận billing event, tính chi phí on-demand và spot, đánh giá autoscaling policy, phân tích waste, tạo dashboard và đo chi phí cho workload GPU thực tế.

Các chiến lược tối ưu hiệu quả nhất trong bài lab là dùng spot instance cho workload có khả năng chịu preemption, dùng mixed precision AMP để giảm thời gian train và chi phí, autoscaling để tránh over-provisioning, theo dõi idle GPU để scale down, và scheduling workload không gấp vào thời điểm chi phí thấp hơn. Trong kết quả thực nghiệm, spot đem lại savings khoảng 70.0%, còn AMP giúp giảm 41.7% chi phí training trên Tesla T4.

Trong ứng dụng thực tế, các kỹ thuật này có thể áp dụng cho project AI/ML như training mô hình computer vision, fine-tuning LLM, batch inference hoặc serving GPU. Một pipeline production nên có monitoring, budget alert, cost allocation theo project/team, checkpointing cho job dài, autoscaling theo utilization thực tế và dashboard để theo dõi cost per experiment. Với các dự án lớn, cần thêm bước forecasting và optimization roadmap để đảm bảo deadline, hiệu năng và ngân sách được cân bằng.

## Trạng thái artifact nộp bài

- Đã có screenshot và chart cho Part 1-8 trong thư mục `screenshots/` và `generated_charts/`.
- Đã có các chart: `finops_cost_breakdown.png`, `finops_timeseries.png`, `real_gpu_comparison.png`, `real_gpu_telemetry.png`, `cost_per_epoch.png`.
- Chưa thấy artifact Part 8.5 theo yêu cầu: `multi_gpu_scaling.png`, `project_forecast.png`, `optimization_roadmap.png`, `advanced_finops_dashboard.png`.
- Chưa thấy screenshot Part 8.5 theo cấu trúc yêu cầu trong thư mục `screenshots/`.
- Lưu ý nhỏ: file screenshot cluster metric hiện tên là `part1_cluster_metric.png`, trong yêu cầu là `part1_cluster_metrics.png` hoặc screenshot Cell 4. Nên đổi tên thống nhất nếu hệ thống chấm kiểm tra theo tên file.
