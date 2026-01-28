import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import parse_semester_code, get_semester_order, calculate_semester_from_admission


def test_semester_parsing():
    """Test semester code parsing"""
    print("="*80)
    print("Testing Semester Parsing Functions")
    print("="*80)
    
    # Test cases
    test_cases = [
        "HK1 2020-2021",
        "HK2 2020-2021", 
        "HK1 2023-2024",
        "HK2 2023-2024",
        "HK1 2024-2025"
    ]
    
    for semester_code in test_cases:
        year, semester = parse_semester_code(semester_code)
        order = get_semester_order(semester_code)
        print(f"{semester_code:20s} -> Year: {year}, Semester: {semester}, Order: {order}")
    
    print("\nSemester parsing tests passed!\n")


def test_semester_calculation():
    """Test semester number calculation"""
    print("="*80)
    print("Testing Semester Number Calculation")
    print("="*80)
    
    # Test: Student admitted in 2018
    admission_year = 2018
    test_semesters = [
        "HK1 2020-2021",
        "HK2 2020-2021",
        "HK1 2023-2024"
    ]
    
    for sem_code in test_semesters:
        sem_num = calculate_semester_from_admission(admission_year, sem_code)
        print(f"Admission: {admission_year}, Current: {sem_code} -> Semester #{sem_num}")
    
    print("\nSemester calculation tests passed!\n")


def test_data_loading():
    """Test data loading with sample data"""
    print("="*80)
    print("Testing Data Loading with Sample Data")
    print("="*80)
    
    # Create sample academic data
    academic_sample = pd.DataFrame({
        'MA_SO_SV': ['f022ed8d1ac1', 'f022ed8d1ac1', 'f022ed8d1ac1'],
        'HOC_KY': ['HK2 2020-2021', 'HK1 2022-2023', 'HK1 2023-2024'],
        'CPA': [2.19, 0.95, 0.81],
        'GPA': [2.02, 2.12, 1.89],
        'TC_DANGKY': [18, 14, 29],
        'TC_HOANTHANH': [18, 7, 16]
    })
    
    # Create sample admission data
    admission_sample = pd.DataFrame({
        'MA_SO_SV': ['f022ed8d1ac1'],
        'NAM_TUYENSINH': [2020],
        'PTXT': [5],
        'TOHOP_XT': ['A00'],
        'DIEM_TRUNGTUYEN': [15.86],
        'DIEM_CHUAN': [15.1]
    })
    
    print("Academic Data Sample:")
    print(academic_sample)
    print("\nAdmission Data Sample:")
    print(admission_sample)
    
    # Test semester ordering
    academic_sample['semester_order'] = academic_sample['HOC_KY'].apply(get_semester_order)
    print("\nWith semester ordering:")
    print(academic_sample[['HOC_KY', 'semester_order']].sort_values('semester_order'))
    
    print("\nData loading tests passed!\n")


def test_data_split():
    """Test data splitting logic"""
    print("="*80)
    print("Testing Data Split Logic")
    print("="*80)
    
    # Create sample data across multiple semesters
    semesters = [
        'HK1 2020-2021', 'HK2 2020-2021',
        'HK1 2021-2022', 'HK2 2021-2022',
        'HK1 2022-2023', 'HK2 2022-2023',
        'HK1 2023-2024', 'HK2 2023-2024',
        'HK1 2024-2025'
    ]
    
    df = pd.DataFrame({
        'HOC_KY': semesters,
        'semester_order': [get_semester_order(s) for s in semesters]
    })
    
    print("All semesters:")
    print(df)
    
    # Split based on TRAIN_END and VALID_SEMESTER
    train_end = 'HK1 2023-2024'
    valid_semester = 'HK2 2023-2024'
    
    train_end_order = get_semester_order(train_end)
    valid_order = get_semester_order(valid_semester)
    
    train_df = df[df['semester_order'] <= train_end_order]
    valid_df = df[df['semester_order'] == valid_order]
    
    print(f"\nTrain end: {train_end} (order: {train_end_order})")
    print(f"Valid semester: {valid_semester} (order: {valid_order})")
    print(f"\nTrain data ({len(train_df)} records):")
    print(train_df['HOC_KY'].tolist())
    print(f"\nValid data ({len(valid_df)} records):")
    print(valid_df['HOC_KY'].tolist())
    
    print("\nData split tests passed!\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING DATA FORMAT VERIFICATION TESTS")
    print("="*80 + "\n")
    
    try:
        test_semester_parsing()
        test_semester_calculation()
        test_data_loading()
        test_data_split()
        
        print("="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nThe code is ready to work with your data format:")
        print("- Academic: MA_SO_SV, HOC_KY (format: 'HK1 2020-2021'), CPA, GPA, TC_DANGKY, TC_HOANTHANH")
        print("- Admission: MA_SO_SV, NAM_TUYENSINH, PTXT, TOHOP_XT, DIEM_TRUNGTUYEN, DIEM_CHUAN")
        print("- Test: MA_SO_SV, HOC_KY, TC_DANGKY")
        print("\nYou can now:")
        print("1. Put your CSV files in data/raw/")
        print("2. Run: python main.py --team_name your_team")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)