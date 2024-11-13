import pytest
from BoostingLRWrapper import BoostingLRWrapper
from utils import start_jvm, stop_jvm, load_dataset_as_Instances, kendalls_tau

# Fixture to start and stop the JVM
@pytest.fixture(scope="session", autouse=True)
def jvm_manager():
    """Start the JVM before tests and stop it after."""
    start_jvm()
    yield
    stop_jvm()

# Fixtures to load the data
@pytest.fixture
def train_data():
    """Fixture to load training data as Weka Instances."""
    return load_dataset_as_Instances("./datasets/sushi_train.xarff")

@pytest.fixture
def test_data():
    """Fixture to load test data as Weka Instances."""
    return load_dataset_as_Instances("./datasets/sushi_test.xarff")

# Initialization Test
def test_initialization():
    """Test the initialization of BoostingLRWrapper."""
    model = BoostingLRWrapper(max_iterations=100, seed=42)  # Provide a default seed
    assert model.max_iterations == 100
    assert model.seed == 42

# Fit Method Test
def test_fit(train_data):
    """Test the fit method of BoostingLRWrapper."""
    model = BoostingLRWrapper(max_iterations=50, seed=42)  # Provide a default seed
    try:
        model.fit(train_data)
    except Exception as e:
        pytest.fail(f"Fit method failed with exception: {e}")

# Predict Method Test
def test_predict(train_data, test_data):
    """Test the predict method of BoostingLRWrapper."""
    model = BoostingLRWrapper(max_iterations=50, seed=42)  # Provide a default seed
    model.fit(train_data)
    predictions = model.predict(test_data)
    
    # Ensure that the number of predictions matches the number of instances
    assert len(predictions) == test_data.numInstances()

# Score Method Test
def test_score(train_data, test_data):
    """Test the score method of BoostingLRWrapper."""
    model = BoostingLRWrapper(max_iterations=50, seed=42)  # Provide a default seed
    model.fit(train_data)
    score = model.score(test_data)
    
    # Check if the score is a float and within a reasonable range
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"

# Additional Tests
def test_kendalls_tau():
    """Test the kendalls_tau utility function."""
    prefs = [1, 2, 3, 4, 5]
    preds = [5, 4, 3, 2, 1]
    tau = kendalls_tau(prefs, preds)
    
    # Use pytest.approx for floating-point comparison
    assert tau == pytest.approx(-1.0, rel=1e-9)
