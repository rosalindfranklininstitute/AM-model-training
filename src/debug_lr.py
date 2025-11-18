from lr import plot_learning_rates

test_csv = "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/test.csv"
all_files_csv = (
    "/ceph/groups/structbio/adaptive_milling_project/2024labels_new/all_files.csv"
)

if __name__ == "__main__":
    plot_learning_rates(
        test_csv, models_to_ignore=["unet", "dynunet", "segresnet", "segresnetds"]
    )
