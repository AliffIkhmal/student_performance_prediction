import re

from model import StudentPerformanceModel


DATASET_REQUIRED_COLUMNS = [
    "StudentID",
    *StudentPerformanceModel.FEATURES,
    "GPA",
    "GradeClass",
]
BATCH_REQUIRED_COLUMNS = list(StudentPerformanceModel.FEATURES)
BATCH_OPTIONAL_COLUMNS = ["StudentID"]
PERIOD_OPTIONAL_COLUMNS = [
    "AcademicSession",
    "Semester",
    "Term",
    "Intake",
    "AcademicYear",
    "RecordDate",
    "CreatedAt",
    "UpdatedAt",
]
ALL_DATASET_COLUMNS = DATASET_REQUIRED_COLUMNS + PERIOD_OPTIONAL_COLUMNS

CANONICAL_COLUMN_LABELS = {
    "StudentID": "Student ID",
    "Age": "Age",
    "Gender": "Gender",
    "ParentalEducation": "Parental Education",
    "StudyTimeWeekly": "Study Time Weekly",
    "Absences": "Absences",
    "ParentalSupport": "Parental Support",
    "Extracurricular": "Extracurricular",
    "Sports": "Sports",
    "Music": "Music",
    "Volunteering": "Volunteering",
    "GPA": "GPA",
    "GradeClass": "Grade Class",
    "AcademicSession": "Academic Session",
    "Semester": "Semester",
    "Term": "Term",
    "Intake": "Intake",
    "AcademicYear": "Academic Year",
    "RecordDate": "Record Date",
    "CreatedAt": "Created Date",
    "UpdatedAt": "Updated Date",
}

COLUMN_ALIAS_MAP = {
    "StudentID": [
        "studentid",
        "studentidentifier",
        "studentnumber",
        "studentno",
        "studentnum",
        "student_id",
        "student_no",
        "student_number",
        "sid",
        "matric",
        "matricno",
        "matricnumber",
        "rollno",
        "rollnumber",
        "id",
    ],
    "Age": ["age", "studentage", "student_age", "years", "yearsold"],
    "Gender": ["gender", "sex", "studentgender", "student_gender"],
    "ParentalEducation": [
        "parentaleducation",
        "parenteducation",
        "parentaleduc",
        "parental_education",
        "parent_education",
        "parent_edu",
        "parentseducation",
        "guardianeducation",
        "guardian_education",
    ],
    "StudyTimeWeekly": [
        "studytimeweekly",
        "studytime",
        "study_time",
        "weeklystudy",
        "weeklystudytime",
        "studyhours",
        "study_hours",
        "studyhrs",
        "studyhoursweekly",
        "studyhoursperweek",
        "studytimeperweek",
        "hoursstudied",
    ],
    "Absences": [
        "absences",
        "absence",
        "absent",
        "absentcount",
        "absencecount",
        "daysabsent",
        "absencedays",
    ],
    "ParentalSupport": [
        "parentalsupport",
        "parental_support",
        "parentsupport",
        "parent_support",
        "guardian_support",
        "familysupport",
    ],
    "Extracurricular": [
        "extracurricular",
        "extra_curricular",
        "extracurriculars",
        "extracurricularactivity",
        "extracurricularactivities",
        "clubs",
    ],
    "Sports": ["sports", "sport", "athletics", "athletic"],
    "Music": ["music", "musical", "musicactivity", "musicactivities"],
    "Volunteering": [
        "volunteering",
        "volunteer",
        "volunteerwork",
        "communityservice",
        "community_service",
    ],
    "GPA": ["gpa", "cgpa", "gradepointaverage", "grade_point_average", "averagegpa"],
    "GradeClass": [
        "gradeclass",
        "grade_class",
        "grade",
        "finalgrade",
        "final_grade",
        "classgrade",
    ],
    "AcademicSession": [
        "academicsession",
        "academic_session",
        "session",
        "schoolsession",
        "school_session",
    ],
    "Semester": ["semester", "sem"],
    "Term": ["term", "academicterm", "academic_term"],
    "Intake": ["intake", "cohort", "admissionbatch", "admission_batch", "entrybatch"],
    "AcademicYear": ["academicyear", "academic_year", "year", "schoolyear", "school_year"],
    "RecordDate": ["recorddate", "record_date", "date", "snapshotdate", "snapshot_date"],
    "CreatedAt": ["createdat", "created_at", "createddate", "created_date", "uploaddate", "upload_date"],
    "UpdatedAt": ["updatedat", "updated_at", "updateddate", "updated_date", "modifiedat", "modified_at"],
}


def normalize_header(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def canonical_candidates(canonical_name):
    candidates = [normalize_header(canonical_name)]
    for alias in COLUMN_ALIAS_MAP.get(canonical_name, []):
        normalized_alias = normalize_header(alias)
        if normalized_alias not in candidates:
            candidates.append(normalized_alias)
    return candidates


def resolve_columns(dataframe, canonical_columns=None, preferred_mapping=None):
    canonical_columns = canonical_columns or ALL_DATASET_COLUMNS
    preferred_mapping = preferred_mapping or {}
    csv_columns = list(dataframe.columns)
    normalized_columns = {
        column_name: normalize_header(column_name)
        for column_name in csv_columns
    }

    auto_mapped = {}
    used_columns = set()

    for canonical_name in canonical_columns:
        preferred_column = preferred_mapping.get(canonical_name)
        if preferred_column in csv_columns and preferred_column not in used_columns:
            auto_mapped[canonical_name] = preferred_column
            used_columns.add(preferred_column)
            continue

        for candidate in canonical_candidates(canonical_name):
            matched_column = next(
                (
                    column_name
                    for column_name in csv_columns
                    if column_name not in used_columns and normalized_columns[column_name] == candidate
                ),
                None,
            )
            if matched_column is None:
                continue

            auto_mapped[canonical_name] = matched_column
            used_columns.add(matched_column)
            break

    return {
        "auto_mapped": auto_mapped,
        "unmapped_canonical": [
            canonical_name
            for canonical_name in canonical_columns
            if canonical_name not in auto_mapped
        ],
        "unmapped_csv": [
            column_name
            for column_name in csv_columns
            if column_name not in used_columns
        ],
        "all_csv_columns": csv_columns,
    }


def apply_column_mapping(dataframe, mapping):
    mapping = mapping or {}
    rename_map = {}

    for canonical_name, actual_name in mapping.items():
        if not actual_name or actual_name not in dataframe.columns:
            continue
        if actual_name == canonical_name:
            continue
        rename_map[actual_name] = canonical_name

    renamed = dataframe.rename(columns=rename_map)
    duplicated = renamed.columns[renamed.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Column mapping creates duplicate canonical names: " + ", ".join(duplicated)
        )

    return renamed


def build_mapping_rows(csv_columns, selected_mapping=None, canonical_columns=None, required_columns=None):
    canonical_columns = canonical_columns or ALL_DATASET_COLUMNS
    required_columns = set(required_columns or DATASET_REQUIRED_COLUMNS)
    selected_mapping = selected_mapping or {}

    return [
        {
            "canonical": canonical_name,
            "label": CANONICAL_COLUMN_LABELS.get(canonical_name, canonical_name),
            "selected_actual": selected_mapping.get(canonical_name, ""),
            "required": canonical_name in required_columns,
            "choices": list(csv_columns),
        }
        for canonical_name in canonical_columns
    ]