
# tests/utils/assertions.py
"""
Custom assertions for skills testing.
"""

def assert_skill_extraction_quality(result, min_skills=5, min_confidence_ratio=0.6):
    """Assert overall quality of skill extraction."""
    assert len(result.skills) >= min_skills, (
        f"Expected at least {min_skills} skills, got {len(result.skills)}"
    )
    
    high_conf_count = len(result.get_high_confidence_skills())
    confidence_ratio = high_conf_count / len(result.skills)
    
    assert confidence_ratio >= min_confidence_ratio, (
        f"Expected {min_confidence_ratio:.1%} high confidence, "
        f"got {confidence_ratio:.1%} ({high_conf_count}/{len(result.skills)})"
    )


def assert_performance_acceptable(result, max_time=5.0):
    """Assert that performance is within acceptable limits."""
    processing_time = result.extraction_metadata.get("processing_time", 0)
    
    assert processing_time <= max_time, (
        f"Processing took {processing_time:.2f}s, should be <= {max_time}s"
    )
    
    assert processing_time > 0, "Processing time should be recorded"


def assert_skill_categories_present(result, expected_categories, min_categories=3):
    """Assert that expected skill categories are present."""
    found_categories = set()
    
    for skill in result.skills:
        found_categories.add(skill.category)
    
    category_overlap = found_categories.intersection(set(expected_categories))
    
    assert len(category_overlap) >= min_categories, (
        f"Expected at least {min_categories} categories from {expected_categories}, "
        f"found {list(category_overlap)}"
    )