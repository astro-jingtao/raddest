import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import shutil

import pytest

from raddest.api.fitter import FlowVI
from raddest.transfer import NegLogPlus


class TestFlowVISaveLoad:

    def test_base_case(self):

        flowvi = FlowVI(transfer=NegLogPlus(),
                        x_min=0,
                        x_max=10,
                        x_err_min=0,
                        x_err_max=10)

        flowvi.save('test_flowvi')
        assert os.path.exists('test_flowvi')

        flowvi_loaded = FlowVI.load('test_flowvi')

        assert flowvi_loaded.x_min == flowvi.x_min
        assert flowvi_loaded.x_max == flowvi.x_max
        assert flowvi_loaded.x_err_min == flowvi.x_err_min
        assert flowvi_loaded.x_err_max == flowvi.x_err_max

        shutil.rmtree('test_flowvi')

    def test_no_flow_case(self):

        flowvi = FlowVI(transfer=NegLogPlus(),
                        x_min=0,
                        x_max=10,
                        x_err_min=0,
                        x_err_max=10)

        flowvi.save('test_flowvi')
        assert os.path.exists('test_flowvi')

        flowvi_loaded = FlowVI.load('test_flowvi')
        assert flowvi_loaded.flow is None

        shutil.rmtree('test_flowvi')

    def test_force_save(self):

        flowvi = FlowVI(transfer=NegLogPlus(),
                        x_min=0,
                        x_max=10,
                        x_err_min=0,
                        x_err_max=10)

        os.makedirs('test_flowvi', exist_ok=True)
        with open('test_flowvi/test_flowvi', 'w', encoding='utf-8') as f:
            f.write('test')

        with pytest.raises(RuntimeError):
            flowvi.save('test_flowvi')

        flowvi.save('test_flowvi', force=True)
        FlowVI.load('test_flowvi')
        shutil.rmtree('test_flowvi')

        with open('test_flowvi', 'w', encoding='utf-8') as f:
            f.write('test')

        with pytest.raises(RuntimeError):
            flowvi.save('test_flowvi')
    
        flowvi.save('test_flowvi', force=True)
        FlowVI.load('test_flowvi')
        shutil.rmtree('test_flowvi')
