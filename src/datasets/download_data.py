# src/download_data.py
from src.utils.download_utils import (
    download_and_extract_from_yadisk,
    #download_and_extract_from_gdrive_file,
)


if __name__ == "__main__":
    DATA_DIR = "data"

    # URLs + filenames for Yandex Datasets
    YADISK_DATASETS = [
        {
            "url": "https://disk.360.yandex.ru/d/8ar2q0chZ2ZePw",
            "filename": "LSRW.zip"
        },
        {
            "url": "https://disk.360.yandex.ru/d/VXSfzoU9gAFu0w",
            "filename": "LOL-v2.zip"
        },
        {
            "url": "https://disk.360.yandex.ru/d/Pv7kHFNq0_Rmpg", 
            "filename": "UHD-LL_test.zip"
        },

        {
            "url": "https://disk.360.yandex.ru/d/jzD-_1KwCYwpYw", 
            "filename": "UHD-LL_train.zip"
        },

        {
            "url": "https://disk.360.yandex.ru/d/kbIRrqEmQG-gBg",
            "filename": "Test_Quality_v2.zip"

        }
        ,
        {
            "url": "https://disk.360.yandex.ru/d/lAauPHGroUrDEQ",
            "filename": "Test_Quality.zip"
            
        }

    ]

    for dataset in YADISK_DATASETS:
        print(f"Downloading and extracting {dataset['filename']} ...")
        download_and_extract_from_yadisk(
            dataset["url"],
            dataset["filename"],
            DATA_DIR
        )

    print("All datasets downloaded and extracted successfully!")