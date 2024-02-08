import threading
import logging
from pathlib import Path
import requests
import gzip
import concurrent

class PDBDownloadManager:
    def __init__(self, list_file_path, pdb_dir, update_threshold=10, save_interval=60):
        self.list_file_path = list_file_path
        self.pdb_dir = pdb_dir
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.failed_pdb = self._load_failed_list()
        self.failed_pdb = self._load_failed_list()
        self.change_counter = 0
        self.update_threshold = update_threshold
        self.save_interval = save_interval
        self.lock = threading.RLock()
        self._start_periodic_save()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def gz(self,pdbID):
        pdbID_lower = pdbID.lower()
        return Path(self.pdb_dir) /f"{pdbID_lower[1:3]}"/ f"{pdbID_lower}.ent.gz"
    
    def check_exists(self, gz_path):
        if not gz_path.exists():
            return False
        elif gz_path.stat().st_size == 0:
            logging.warning(f"File is empty: {gz_path}")
            return False
        elif gz_path.is_dir():
            logging.error(f"Expected a file but found a directory: {gz_path}")
            return False
        elif gz_path.stat().st_size == 0:
            logging.warning(f"File is empty: {gz_path}")
            return False
        else:
            try:
                with gz_path.open('rb') as f:
                    # Try reading the first byte to check if it's readable and not corrupted
                    gzip.GzipFile(fileobj=f).read(1)
                logging.info(f"File exists, is not empty, and is readable: {gz_path}")
                return True
            except (OSError, gzip.BadGzipFile):
                logging.warning(f"File is not readable or is corrupted: {gz_path}")
                return False
        
    def download_pdb(self, pdbID, max_retries=3, failed_retry=1):
        pdbID_lower=pdbID.lower()
        gz_path = self.gz(pdbID)
        if self.check_exists(gz_path):
            return True
        download_url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdbID_lower[1:3]}/pdb{pdbID_lower}.ent.gz"
        
        if pdbID_lower in self.failed_pdb:
            max_retries=failed_retry

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(download_url, stream=True, timeout=10)
                response.raise_for_status()

                gz_prepath = self.pdb_dir / "{pdbID_lower[1:3]}"
                gz_prepath.mkdir(parents=True, exist_ok=True)

                with open(gz_path, 'wb') as f:
                    f.write(response.content)

                logging.info(f"Downloaded and saved: {gz_path}")
                return True

            except Exception as e:
                logging.warning(f"Retry {retry_count + 1} for {pdbID_lower.upper()}: {e}")
                retry_count += 1

        logging.error(f"Failed to download after {max_retries} retr{'y' if max_retries==1 else 'ies'}: {pdbID_lower.upper()}")
        self.add_failed_pdb(pdbID_lower)
        return False
    
    def download_pdbs(self, pdbs):
        # Remove duplicates and sort
        pdbs = sorted(set(pdbs))

        download_results = {pdb:False for pdb in pdbs}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_pdbID = {executor.submit(self.download_pdb, pdb): pdb for pdb in pdbs}

            for future in concurrent.futures.as_completed(future_to_pdbID):
                pdbID = future_to_pdbID[future]
                try:
                    success = future.result()
                    download_results[pdbID] = success
                except Exception as exc:
                    logging.error(f"{pdbID} generated an exception: {exc}")
                    download_results[pdbID] = False

        return download_results
        
    def _load_failed_list(self):
        try:
            if self.list_file_path.exists():
                with self.list_file_path.open() as f:
                    return set(line.strip() for line in f)
            return set()
        except Exception as e:
            logging.error(f"Error loading failed list: {e}")
            return set()

    def add_failed_pdb(self, pdb_id):
        with self.lock:
            self.failed_pdb.add(pdb_id)
            self._increment_counter()

    def remove_failed_pdb(self, pdb_id):
        with self.lock:
            self.failed_pdb.discard(pdb_id)
            self._increment_counter()

    def _increment_counter(self):
        self.change_counter += 1
        if self.change_counter >= self.update_threshold:
            self._update_failed_list_file()

    def _update_failed_list_file(self):
        with self.lock:
            try:
                # Create a temporary file in the same directory as the target file
                temp_file_path = self.list_file_path.with_suffix('.tmp')
                with temp_file_path.open('w') as temp_file:
                    for pdb_id in self.failed_pdb:
                        temp_file.write(pdb_id + '\n')
                
                # Atomically rename the temporary file to the target file
                temp_file_path.rename(self.list_file_path)
                logging.info("Failed PDB list updated.")
            except Exception as e:
                logging.error(f"Error updating failed list file: {e}")

    def finalize_updates(self):
        with self.lock:
            self._update_failed_list_file()

    def _start_periodic_save(self):
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()

    def _periodic_save(self):
        while True:
            time.sleep(self.save_interval)
            with self.lock:
                self._update_failed_list_file()


import pytest
import threading
import time
import pytest
from unittest.mock import Mock

#from pdb_download_manager import PDBDownloadManager  # Adjust the import as per your project structure

@pytest.fixture
def pdb_manager(tmp_path):
    return PDBDownloadManager(tmp_path / "test_failed_pdb_list.txt", tmp_path/'pdb_dir')

@pytest.fixture
def mock_get(mocker):
    mock = Mock()
    mocker.patch('requests.get', return_value=mock)
    return mock

def test_add_failed_pdb(pdb_manager):
    pdb_manager.add_failed_pdb("1ABC")
    assert "1ABC" in pdb_manager.failed_pdb

def test_remove_failed_pdb(pdb_manager):
    pdb_manager.add_failed_pdb("1ABC")
    pdb_manager.remove_failed_pdb("1ABC")
    assert "1ABC" not in pdb_manager.failed_pdb

def test_file_update(pdb_manager):
    pdb_manager.add_failed_pdb("1ABC")
    pdb_manager.finalize_updates()
    with open(pdb_manager.list_file_path, 'r') as file:
        content = file.read()
    assert "1ABC" in content

def test_thread_safety(pdb_manager):
    # Enhanced to test a mix of add and remove operations
    pdb_manager.update_threshold=5
    threads = []

    for i in range(10):
        t_add = threading.Thread(target=pdb_manager.add_failed_pdb, args=(f"{i}ABC",))
        t_remove = threading.Thread(target=pdb_manager.remove_failed_pdb, args=(f"{i}ABC",))
        threads.extend([t_add, t_remove])
        t_add.start()
        t_remove.start()

    for t in threads:
        t.join()

    with open(pdb_manager.list_file_path, 'r') as file:
        content = file.readlines()
    assert len(content) <= 10  # Check if the file has a reasonable number of entries

def test_atomic_file_writing(pdb_manager):
    # Test to simulate interruption during file write
    #manager = PDBDownloadManager(test_file, update_threshold=1)
    # Simulate an interruption here, if possible
    # Verify that the file is either fully written or not written at all
    pass

def test_periodic_save(pdb_manager):
    pdb_manager.update_threshold=100
    pdb_manager.save_interval=0.2
    pdb_manager.add_failed_pdb("1ABC")
    time.sleep(0.4)  # Wait for the periodic save to trigger
    with open(pdb_manager.list_file_path, 'r') as file:
        content = file.read()
    assert "1ABC" in content

#from pdb_download_manager import PDBDownloadManager  # Adjust the import as per your project structure

def test_download_single_pdb(mock_get, pdb_manager):
    mock_get.return_value.ok = True
    mock_get.return_value.content = b'PDB data'

    pdb_id = '1ABC'
    sucess = pdb_manager.download_pdb(pdb_id)
    assert sucess
    assert pdb_manager.gz(pdb_id).exists()

def test_download_multiple_pdbs(mock_get, pdb_manager):
    pdbs = ['1ABC', '2XYZ', '3DEF']
    mock_get.return_value.ok = True
    mock_get.return_value.content = b'PDB data'

    sucess = pdb_manager.download_pdbs(pdbs)

    for pdb_id in pdbs:
        assert sucess[pdb_id]
        assert pdb_manager.gz(pdb_id).exists()
