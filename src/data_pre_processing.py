import json
import logging
import os
import pdb
from pathlib import Path

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__file__)

# option + shift + f to format
# option + shift + o to organise imports
# dpb.set_trace() to set trace for debugging


def main():
    data_path = Path(__file__).parent.parent / "EEG_data"

    assert data_path.exists(), "Data path does not exist"

    dirs = os.listdir(data_path / "ds005207")
    subs = [d for d in dirs if d.startswith("sub")]

    for sub in subs:
        sub_root = data_path / "ds005207" / sub / "ses-001" / "eeg"
        eeg_file = sub_root / f"{sub}_ses-001_task-sleep_acq-PSG_eeg.set"
        scoring_fname = sub_root / f"{sub}_ses-001_task-sleep_acq-PSGScoring_events.tsv"
        mapping_fname = (
            data_path / "ds005207" / "task-sleep_acq-cEEGridScoring_events.json"
        )
        # Make output path folder and check that it didnt fail quietly
        output_path = Path(__file__).parent.parent / "EEG_data" / "cleaned_data"
        try:
            os.mkdir(output_path)
        except FileExistsError:
            print(f"Directory '{output_path}' failed")

        logger.debug("PATH FOR SUBJECT DATA", sub)

        # Error checking to make sure the file paths exist
        assert sub_root.exists(), "Subject root does not exist"
        assert eeg_file.exists(), "EEG file file does not exist"
        assert scoring_fname.exists(), "Secoring file does not exist"
        assert mapping_fname.exists(), "Mapping file does not exist"
        assert output_path.exists(), "Output file does not exist"

        # --------------------------------
        # --- Main Proccessing of Data ---
        # --------------------------------

        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

        with open(mapping_fname, "r") as f:
            eeg_json = json.load(f)

        sleep_stage_mapping = eeg_json["staging"]["Levels"]
        logger.debug("Sleep Stage Mapping:")
        logger.debug(sleep_stage_mapping)

        scoring_df = pd.read_csv(scoring_fname, sep="\t")
        logger.debug("\nScoring file preview:")
        logger.debug(scoring_df.head())

        onsets = scoring_df["onset"].to_numpy()
        durations = np.full(len(scoring_df), 30)
        descriptions = (
            scoring_df["staging"].map(str).map(sleep_stage_mapping).to_numpy()
        )

        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )

        raw.set_annotations(annotations)

        events, event_id = mne.events_from_annotations(raw, event_id=None)

        logger.debug("\nEvent ID dictionary created by MNE:")
        logger.debug(event_id)

        logger.debug("Channels in raw data", sub, "with names ", raw.ch_names)
        # #Piece of code that removes a ll but the EEG data 
        # #'F4:A1', 'C4:A1', 'O1:A2', 
        # channels_to_keep = [
        # 'F4:M1', 'C4:M1', 'O1:M2', 
        # 'F3:M2', 'C3:M2', 'O2:M1'


        channels_to_drop
        raw.pick(channels_to_keep)
        logger.debug("channels included in final data", raw.ch_names)

        # --------------------------------
        # ---- Actual Data Cleaning  -----
        # --------------------------------

        # Band and notch filter applied
        raw.filter(l_freq=0.5, h_freq=100, fir_design="firwin")
        raw.notch_filter(freqs=50, fir_design="firwin")

        # ----------------------------------------------------------------
        # ---- Parititioning cleaned data into 30 sec Epoch Objects  -----
        # ----------------------------------------------------------------
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=event_id,
            tmin=0,
            tmax=30,
            preload=True, 
            baseline=None,
        )

        logger.info("\nCreated epochs object:")
        logger.info(epochs)

        logger.info("Epochs made:", len(epochs))

        cleaned_epochs_fname = os.path.join(output_path, f"{sub}_cleaned-epo.fif")
        epochs.save(cleaned_epochs_fname, overwrite=True)
        logger.info(f"\nSuccessfully saved cleaned epochs to:\n{cleaned_epochs_fname}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    main()
