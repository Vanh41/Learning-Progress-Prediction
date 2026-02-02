# 🏗️ Feature Engineering Documentation

Class `FeatureEngineer` chịu trách nhiệm trích xuất đặc trưng từ dữ liệu lịch sử sinh viên. Quy trình này tập trung vào việc mô hình hóa **Năng lực (Ability)**, **Xu hướng (Trend)** và **Hành vi rủi ro (Risk Behavior)**.

## ⚙️ Preprocessing Logic

Trước khi tạo feature, dữ liệu được xử lý như sau để đảm bảo tính toàn vẹn của chuỗi thời gian (Time-series integrity):

1. **Sorting:** `sort_values(['MA_SO_SV', 'semester_order'])` $\rightarrow$ Đảm bảo đúng thứ tự thời gian.
2. **Lagging:** Dùng `shift(1)` cho tất cả các biến lịch sử để ngăn chặn **Data Leakage** (Không dùng tương lai dự báo quá khứ).

---

## 📊 Chi tiết các nhóm Feature

### 1. Admission Features (Thông tin đầu vào)

*Đánh giá xuất phát điểm và giai đoạn đào tạo hiện tại.*

| Tên biến (Feature) | Logic / Công thức                  | Insight (Ý nghĩa nghiệp vụ)                                                                                                                   |
| :------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `diem_vuot_chuan`  | `DIEM_TRUNGTUYEN` - `DIEM_CHUAN` | **Vị thế đầu vào.** Sinh viên có điểm vượt chuẩn cao thường có nền tảng tốt và ít rủi ro hơn sinh viên đậu "vớt". |
| `nam_tuoi`         | `Current_Year` - `NAM_TUYENSINH` | **Độ tuổi.** Tuổi cao hơn so với khóa học có thể ám chỉ việc học lại, đi làm thêm hoặc gián đoạn học tập.           |
| `semester_number`  | `cumcount() + 1`                   | **Giai đoạn.** Hành vi đăng ký và rủi ro rớt môn thay đổi theo năm học (Năm 1 bỡ ngỡ vs Năm 4 môn khó).                 |
| `is_freshman`      | `Prev_TC_DANGKY == 0`              | **Cờ Tân sinh viên.** Đánh dấu các quan sát chưa có lịch sử học tập (Cold-start).                                             |

### 2. History Features (Lịch sử học tập)

*Phản ánh năng lực và thói quen gần nhất (Short-term memory).*

| Tên biến (Feature)  | Logic / Công thức                                                          | Insight (Ý nghĩa nghiệp vụ)                                                                                                                                   |
| :-------------------- | :--------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Prev_GPA`          | GPA kỳ trước (`shift(1)`)                                               | **Hiệu suất gần nhất.** Dự báo tốt nhất cho kết quả kỳ này chính là kết quả của kỳ liền trước.                                         |
| `Prev_CPA`          | CPA tích lũy kỳ trước                                                   | **Sức học dài hạn.** Phản ánh năng lực gốc của sinh viên.                                                                                        |
| `prev_gpa_cpa_diff` | `Prev_GPA` - `Prev_CPA`                                                  | **Đà phong độ (Momentum).** `<br>` (+) Đang tiến bộ vượt bậc so với chính mình. `<br>` (-) Đang sa sút phong độ.                       |
| `load_factor`       | $\frac{\text{TC Đăng ký kỳ này}}{\text{Sức học trung bình 5 kỳ}}$ | **Chỉ số quá tải (Burnout Risk).** Nếu > 1.0: Sinh viên đang đăng ký vượt quá năng lực lịch sử của họ $\rightarrow$ Nguy cơ rớt cao. |
| `failed_last_sem`   | `Prev_HOANTHANH` < `Prev_DANGKY`                                         | **Cú sốc tâm lý.** Cờ báo hiệu sinh viên vừa gặp thất bại ở kỳ trước.                                                                       |

### 3. Trend Features (Xu hướng & Tích lũy)

*Mô hình hóa sự biến động theo thời gian.*

| Tên biến (Feature)       | Logic / Công thức                                            | Insight (Ý nghĩa nghiệp vụ)                                                                                                             |
| :------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `gpa_trend_slope`        | Linear Regression Slope (3 kỳ)                                | **Hướng đi của điểm số.** `<br>` `>0`: Điểm đang cải thiện dần. `<br>` `<0`: Điểm đang lao dốc không phanh. |
| `gpa_volatility`         | Rolling Std Dev (4 kỳ)                                        | **Độ ổn định.** Sinh viên có điểm số trồi sụt thất thường khó dự đoán hơn và rủi ro hơn.                       |
| `accumulated_fail_ratio` | $\frac{\sum \text{Credits Failed}}{\sum \text{Credits Reg}}$ | **Gánh nặng nợ nần.** Tỷ lệ nợ môn tích lũy càng cao, áp lực tâm lý và nguy cơ bỏ học càng lớn.                  |
| `credit_velocity`        | $\frac{\text{Tổng TC Đạt}}{\text{Số kỳ đã học}}$     | **Tốc độ ra trường.** Tốc độ trung bình thấp báo hiệu nguy cơ ra trường muộn.                                         |

### 4. Risk Features (Hành vi rủi ro cao)

*Các mẫu hành vi đặc biệt báo hiệu nguy hiểm.*

| Tên biến (Feature)              | Logic / Công thức                                                             | Insight (Ý nghĩa nghiệp vụ)                                                                                                                                                                                       |
| :-------------------------------- | :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`aggressive_recovery`** | `failed_last_sem` **AND** `<br>` (`TC_DANGKY` > `Prev_TC_DANGKY`) | **Hành vi "Gỡ gạc" (Gambling).** `<br>` Sinh viên vừa rớt môn nhưng lại đăng ký **nhiều tín chỉ hơn** để gỡ lại nhanh. Đây là hành vi cực kỳ rủi ro dẫn đến "gãy" tiếp. |
| `expected_real_credits`         | `TC_DANGKY` * $(1 - \text{Fail Ratio})$                                     | **Kỳ vọng thực tế.** Điều chỉnh con số đăng ký ảo về con số thực tế có thể đạt được dựa trên lịch sử rớt môn.                                                                      |

---

> **Note:**
>
> * Các biến `Category` (Vùng miền, Khoa viện) được giữ nguyên dạng chuỗi để xử lý bằng CatBoost/Encoding sau này.
> * Các giá trị `NaN` sinh ra do Lagging được điền bằng `-1` hoặc `0` để phân biệt với dữ liệu thực.
>
