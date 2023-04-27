import os
from sdv import Metadata

def generate_metadata(dataset_name, original_data, save_metadata=False):
    if dataset_name == "biodegradability":
        return generate_biodegradability_metadata(dataset_name, original_data, save_metadata)

def save_metadata(metadata, dataset_name):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/metadata'

    metadata.to_json(f'{path}/{dataset_name}_metadata.json')

def generate_biodegradability_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()
    metadata.add_table(
        name="molecule",
        data=original_data["molecule"],
        primary_key="molecule_id",
    )

    metadata.add_table(
        name="atom",
        data=original_data["atom"],
        primary_key="atom_id",
    )

    metadata.add_table(
        name="bond",
        data=original_data["bond"],
    )

    metadata.add_table(
        name="gmember",
        data=original_data["gmember"],
    )

    metadata.add_table(
        name="group",
        data=original_data["group"],
        primary_key="group_id",
    )

    metadata.add_relationship(
        parent="molecule",
        child="atom",
        foreign_key="molecule_id",
    )

    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom_id",
    )
    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom_id2",
    )

    metadata.add_relationship(
        parent="atom",
        child="gmember",
        foreign_key="atom_id",
    )

    metadata.add_relationship(
        parent="group",
        child="gmember",
        foreign_key="group_id",
    )
    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata