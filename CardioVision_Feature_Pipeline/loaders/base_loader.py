import abc

class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def load_record(self, file_path, dataset_source):
        """
        Loads an ECG record.
        Returns a dictionary with:
        - ecg_id: str
        - patient_id: str
        - dataset_source: str
        - signal: np.ndarray shape (num_leads, num_samples)
        - fs: int
        - labels: list of str
        """
        pass
