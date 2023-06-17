import os
import unittest
import pandas as pd
from src.download_load_data import download, load_as_dataframe

class TestDownloadLoadData(unittest.TestCase):
    
    def setUp(self):
        self.url = "https://drive.google.com/file/d/1mMWplFXdc3sROA4eqr5yB-zsVWpAikfx/view?usp=sharing"
        self.filename = 'test_particle_data.h5'
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', self.filename)

    def test_download(self):
        download(self.url, self.filename)
        self.assertTrue(os.path.exists(self.file_path))

    def test_load_as_dataframe(self):
        if not os.path.exists(self.file_path):
            download(self.url, self.filename)
        data_frame = load_as_dataframe(self.file_path)
        self.assertIsInstance(data_frame, pd.DataFrame)
        self.assertGreater(len(data_frame), 0)

if __name__ == "__main__":
    unittest.main()
