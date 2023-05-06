import os
from sdv import Metadata


def generate_metadata(dataset_name, original_data, save_metadata=False):
    if dataset_name == "biodegradability":
        return generate_biodegradability_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "rossmann-store-sales":
        return generate_rossman_metadata(dataset_name, original_data, save_metadata)
    raise ValueError(f"Dataset {dataset_name} not supported")


def save_metadata(metadata, dataset_name):
    cwd = os.getcwd()
    cwd_project = cwd.split(
        'Synthetic-data-generation-project')[0] + 'Synthetic-data-generation-project'
    path = cwd_project + '/data/metadata'

    metadata.to_json(f'{path}/{dataset_name}_metadata.json')


def generate_rossman_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()
    metadata.add_table(
        name='store',
        data=original_data['store'],
        primary_key='Store',
    )

    metadata.add_table(
        name='test',
        data=original_data['test'],
        primary_key='Id',
        fields_metadata={
            'Date': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
        }
    )

    metadata.add_relationship(
        parent='store',
        child='test',
        foreign_key='Store',
    )

    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata


def generate_biodegradability_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()
    metadata.add_table(
        name="molecule",
        data=original_data["molecule"],
        primary_key="molecule_id",
        fields_metadata={
            'molecule_id': {
                'type': 'categorical',
            },
        }
    )

    metadata.add_table(
        name="atom",
        data=original_data["atom"],
        primary_key="atom_id",
        fields_metadata={
            'atom_id': {
                'type': 'categorical',
            },
        }
    )

    metadata.add_table(
        name="bond",
        data=original_data["bond"],
        fields_metadata={
            'atom_id': {
                'type': 'categorical',
            },
            'atom_id2': {
                'type': 'categorical',
            },
        }
    )

    metadata.add_table(
        name="gmember",
        data=original_data["gmember"],
        fields_metadata={
            'atom_id': {
                'type': 'categorical',
            },
        }
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
