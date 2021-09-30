import pytest
import copy

@pytest.fixture(scope="module")
def the_list():
    return [1, 2, 3]

@pytest.fixture(scope="module")
def copied_lists(the_list):
    print("called")
    # assertion can be inserted inside fixutre
    assert the_list[0] == 1
    copied = [copy.copy(the_list) for _ in range(3)]
    return copied

def test_copied_lists(copied_lists):
    assert len(copied_lists) == 3
    assert len(copied_lists[0]) == 3

def test_dummy_check_fixture_scope(copied_lists):
    assert False
