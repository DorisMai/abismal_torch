import pytest

def base_test_setters(setter_func, ds, length):
    assert ds._tensor_data is None
    ds[1]
    #Test that _tensor_data is populated
    assert ds._tensor_data is not None
    setter_func(ds)
    #Test that _tensor_data is gone
    assert ds._tensor_data is None
    for batch in ds:
        pass
    assert ds._tensor_data is not None
    assert len(ds._image_data)== length
    return ds

def base_test_dataset(ds, length, rasu_id):
    assert len(ds._image_data) == 0
    assert ds._tensor_data is None
    assert len(ds) == length
    #assert ds._tensor_data is None
    assert len(ds._image_data) == 0
    for i,batch in enumerate(ds):
        assert batch["image_id"].ndim == 1
        assert batch["rasu_id"].ndim == 1
        assert batch["hkl_in"].ndim == 2
        assert batch["hkl_in"].shape[-1] == 3
        assert batch["resolution"].ndim == 1
        assert batch["wavelength"].ndim == 1
        assert batch["metadata"].ndim == 2
        assert batch["iobs"].ndim == 1
        assert batch["sigiobs"].ndim == 1
        assert all(batch['image_id'] == i)
        assert all(batch['rasu_id'] == rasu_id)

    for k,v in ds._tensor_data.items():
        assert len(v) == 0
    assert len(ds._image_data) == length

    ds[0]
    ds[length - 1]
    ds[-length]

@pytest.mark.parametrize('file_type', 
    [
        'stills', 
        pytest.param('mtz', marks=pytest.mark.xfail(reason="Mtz does not have lazy length")),
])
def test_lazy_len(file_type, get_dataset):
    ds = get_dataset(file_type)
    len(ds)
    assert ds._tensor_data is None

@pytest.mark.parametrize('rasu_id', [0, 1, 5])
@pytest.mark.parametrize('file_type', ['stills', 'mtz'])
def test_dataset(file_type, num_dials_stills, rasu_id, get_dataset):
    length = num_dials_stills
    ds = get_dataset(file_type, rasu_id=rasu_id)
    base_test_dataset(ds, length, rasu_id)

@pytest.mark.parametrize('file_type', ['stills', 'mtz'])
def test_stills_dataset_cell_assignment(file_type, num_dials_stills, get_dataset):
    def f(ds):
        ds.cell = [34., 45., 98., 90., 90., 90.]
    ds = get_dataset(file_type)
    base_test_setters(f, ds, num_dials_stills)

@pytest.mark.parametrize('file_type', ['stills', 'mtz'])
def test_stills_dataset_spacegroup_assignment(file_type, num_dials_stills, get_dataset):
    def f(ds):
        ds.spacegroup = 19
    ds = get_dataset(file_type)
    base_test_setters(f, ds, num_dials_stills)

def test_stills_dataset_expt_assignment(num_dials_stills, get_dataset):
    def f(ds):
        ds.expt_file = ds.expt_file
    ds = get_dataset('stills')
    base_test_setters(f, ds, num_dials_stills)

def test_stills_dataset_refl_assignment(num_dials_stills, get_dataset):
    def f(ds):
        ds.refl_file = ds.refl_file
    ds = get_dataset('stills')
    base_test_setters(f, ds, num_dials_stills)

def test_mtz_dataset_mtz_assignment(num_dials_stills, get_dataset):
    def f(ds):
        ds.mtz_file = ds.mtz_file
    ds = get_dataset('mtz')
    base_test_setters(f, ds, num_dials_stills)

@pytest.mark.parametrize('file_type', ['stills'])
def test_stills_dataset_refl_assignment(file_type, num_dials_stills, get_dataset):
    def f(ds):
        ds.refl_file = ds.refl_file
    ds = get_dataset(file_type)
    base_test_setters(f, ds, num_dials_stills)

@pytest.mark.parametrize('dmin', [1., 3., 5.])
@pytest.mark.parametrize('file_type', ['stills'])
def test_stills_dataset_dmin_assignment(file_type, num_dials_stills, dmin, get_dataset):
    def f(ds):
        ds.dmin = dmin
    ds = get_dataset(file_type)
    base_test_setters(f, ds, num_dials_stills)
    for batch in ds:
        assert batch['resolution'].min() >= dmin

