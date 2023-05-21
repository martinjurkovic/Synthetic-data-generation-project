import os
from sdv import Metadata

def get_root_table(dataset_name):
    if dataset_name == "biodegradability":
        return "molecule"
    if dataset_name == "rossmann-store-sales":
        return "store"
    if dataset_name == "mutagenesis":
        return "molecule"
    if dataset_name == "coupon-purchase-prediction":
        return "user_list"
    if dataset_name == "telstra-competition-dataset":
        return "train"
    if dataset_name == "zurich":
        return "customers"
    raise ValueError(f"Dataset {dataset_name} not supported")


def generate_metadata(dataset_name, original_data, save_metadata=False):
    if dataset_name == "biodegradability":
        return generate_biodegradability_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "rossmann-store-sales":
        return generate_rossman_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "mutagenesis":
        return generate_mutagenesis_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "coupon-purchase-prediction":
        return generate_coupon_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "telstra-competition-dataset":
        return generate_telstra_metadata(dataset_name, original_data, save_metadata)
    if dataset_name == "zurich":
        return generate_zurich_metadata(dataset_name, original_data, save_metadata)
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


def generate_mutagenesis_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()
    atom_fields = {
        'atom_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'molecule_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'element': {
            'type': 'categorical'
        },
        'type': {
            'type': 'numerical',
            'subtype': 'integer'
        },
        'charge': {
            'type': 'numerical',
            'subtype': 'float',
        },
    }

    bond_fields = {
        'bond_id': {
            'type': 'id',
            'subtype': 'integer'
        },
        'atom1_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'atom2_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'type': {
            'type': 'numerical',
            'subtype': 'integer'
        }
    }

    molecule_fields = {
        'molecule_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'ind1': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'inda': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'logp': {
            'type': 'numerical',
            'subtype': 'float'
        },
        'lumo': {
            'type': 'numerical',
            'subtype': 'float'
        },
        'mutagenic': {
            'type': 'categorical'
        },

    }

    metadata.add_table(
        name="molecule",
        data=original_data["molecule"],
        primary_key="molecule_id",
        fields_metadata=molecule_fields
    )

    metadata.add_table(
        name="atom",
        data=original_data["atom"],
        primary_key="atom_id",
        fields_metadata=atom_fields
    )

    metadata.add_table(
        name="bond",
        data=original_data["bond"],
        primary_key="bond_id",
        fields_metadata=bond_fields
    )

    metadata.add_relationship(
        parent="molecule",
        child="atom",
        foreign_key="molecule_id",
    )

    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom1_id",
    )
    metadata.add_relationship(
        parent="atom",
        child="bond",
        foreign_key="atom2_id",
    )

    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata


def generate_biodegradability_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()

    # Specification of fields propreties
    atom_fields = {
        'atom_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'molecule_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'type': {
            'type': 'categorical'
        }
    }

    bond_fields = {
        'bond_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'atom_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'atom_id2': {
            'type': 'id',
            'subtype': 'string'
        },
        'type': {
            'type': 'categorical'
        }
    }

    molecule_fields = {
        'molecule_id': {
            'type': 'id',
            'subtype': 'string'
        },
        'activity': {
            'type': 'numerical',
            'subtype': 'float'
        },
        'logp': {
            'type': 'numerical',
            'subtype': 'float'
        },
        'mweight': {
            'type': 'numerical',
            'subtype': 'float'
        },

    }

    metadata.add_table(
        name="molecule",
        data=original_data["molecule"],
        primary_key="molecule_id",
        fields_metadata=molecule_fields
    )

    metadata.add_table(
        name="atom",
        data=original_data["atom"],
        primary_key="atom_id",
        fields_metadata=atom_fields
    )

    metadata.add_table(
        name="bond",
        data=original_data["bond"],
        primary_key="bond_id",
        fields_metadata=bond_fields
    )

    # metadata.add_table(
    #     name="gmember",
    #     data=original_data["gmember"],
    #     primary_key="gmember_id",
    #     fields_metadata={
    #         'atom_id': {
    #             'type': 'categorical',
    #         },
    #     }
    # )

    # metadata.add_table(
    #     name="group",
    #     data=original_data["group"],
    #     primary_key="group_id",
    # )

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

    # metadata.add_relationship(
    #     parent="atom",
    #     child="gmember",
    #     foreign_key="atom_id",
    # )

    # metadata.add_relationship(
    #     parent="group",
    #     child="gmember",
    #     foreign_key="group_id",
    # )
    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata


def generate_telstra_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()

    train_fields = {
        'id': {
            'type': 'id',
            'subtype': 'string',
        },
        'location': {
            'type': 'categorical',
        },
        'fault_severity': {
            'type': 'categorical',
        },
    }

    severity_type_fields = {
        'id': {
            'type': 'id',
            'subtype': 'string',
        },
        'severity_type_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'severity_type': {
            'type': 'categorical',
        },
    }

    event_type_fields = {
        'id': {
            'type': 'id',
            'subtype': 'string',
        },
        'event_type_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'event_type': {
            'type': 'categorical',
        },
    }

    log_feature_fields = {
        'id': {
            'type': 'id',
            'subtype': 'string',
        },
        'log_feature_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'log_feature': {
            'type': 'categorical',
        },
        'volume': {
            'type': 'numerical',
            'subtype': 'integer',
        },
    }

    resource_type_fields = {
        'id': {
            'type': 'id',
            'subtype': 'string',
        },
        'resource_type_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'resource_type': {
            'type': 'categorical',
        },
    }

    metadata.add_table(
        name="train",
        data=original_data["train"],
        primary_key="id",
        fields_metadata=train_fields
    )

    metadata.add_table(
        name="severity_type",
        data=original_data["severity_type"],
        primary_key="severity_type_id",
        fields_metadata=severity_type_fields
    )

    metadata.add_table(
        name="event_type",
        data=original_data["event_type"],
        primary_key="event_type_id",
        fields_metadata=event_type_fields
    )

    metadata.add_table(
        name="log_feature",
        data=original_data["log_feature"],
        primary_key="log_feature_id",
        fields_metadata=log_feature_fields
    )

    metadata.add_table(
        name="resource_type",
        data=original_data["resource_type"],
        primary_key="resource_type_id",
        fields_metadata=resource_type_fields
    )

    metadata.add_relationship(
        parent="train",
        child="severity_type",
        foreign_key="id",
    )

    metadata.add_relationship(
        parent="train",
        child="event_type",
        foreign_key="id",
    )

    metadata.add_relationship(
        parent="train",
        child="log_feature",
        foreign_key="id",
    )

    metadata.add_relationship(
        parent="train",
        child="resource_type",
        foreign_key="id",
    )

    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata


def generate_coupon_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()

    user_list_fields = {
        'USER_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'REG_DATE': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'SEX_ID': {
            'type': 'categorical',
        },
        'AGE': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'WITHDRAW_DATE': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'PREF_NAME': {
            'type': 'categorical',
        },
    }

    coupon_visit_train_fields = {
        'coupon_visit_train_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'PURCHASE_FLG': {
            'type': 'categorical',
        },
        'PURCHASEID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'I_DATE': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'PAGE_SERIAL': {
            'type': 'categorical',
        },
        'REFERRER_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'VIEW_COUPON_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'USER_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'SESSION_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
    }

    coupon_detail_train_fields = {
        'coupon_detail_train_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'ITEM_COUNT': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'I_DATE': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'SMALL_AREA_NAME': {
            'type': 'categorical',
        },
        'PURCHASEID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'USER_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
        'COUPON_ID_hash': {
            'type': 'id',
            'subtype': 'string',
        },
    }

    metadata.add_table(
        name="user_list",
        data=original_data["user_list"],
        primary_key="USER_ID_hash",
        fields_metadata=user_list_fields
    )

    metadata.add_table(
        name="coupon_visit_train",
        data=original_data["coupon_visit_train"],
        primary_key="coupon_visit_train_id",
        fields_metadata=coupon_visit_train_fields
    )

    metadata.add_table(
        name="coupon_detail_train",
        data=original_data["coupon_detail_train"],
        primary_key="coupon_detail_train_id",
        fields_metadata=coupon_detail_train_fields
    )

    metadata.add_relationship(
        parent="user_list",
        child="coupon_visit_train",
        foreign_key="USER_ID_hash",
    )

    metadata.add_relationship(
        parent="user_list",
        child="coupon_detail_train",
        foreign_key="USER_ID_hash",
    )

    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata


def generate_zurich_metadata(dataset_name, original_data, save_metadata=False):
    metadata = Metadata()
    # customer data:
    # customer_id,customer_type,gender,country_part,date_of_birth(1986-02-17),household_id,household_role
    customer_fields = {
        'customer_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'customer_type': {
            'type': 'categorical',
        },
        'gender': {
            'type': 'categorical',
        },
        'country_part': {
            'type': 'categorical',
        },
        'date_of_birth': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'household_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'household_role': {
            'type': 'categorical',
        },
    }

    # policy data:
    # underwriting_date (2013-05-17), first_end_date (2013-07-14),cancellation_or_end_date (2013-07-14),
    # policy_id,sales_channel,customer_id,premium,status,line,product_name,product_group
    policy_fields = {
        'underwriting_date': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'first_end_date': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'cancellation_or_end_date': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'policy_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'sales_channel': {
            'type': 'categorical',
        },
        'customer_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'premium': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'status': {
            'type': 'categorical',
        },
        'line': {
            'type': 'categorical',
        },
        'product_name': {
            'type': 'categorical',
        },
        'product_group': {
            'type': 'categorical',
        },
    }

    # claim data:
    # claim_expense,claim_paid,claim_recovered,claim_reserved,claim_status,claim_total_value,
    # date_closed (2013-05-17),date_open (2013-05-17),event_date (2013-05-17),policy_id,claim_id,customer_id

    claim_fields = {
        'claim_expense': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'claim_paid': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'claim_recovered': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'claim_reserved': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'claim_status': {
            'type': 'categorical',
        },
        'claim_total_value': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'date_closed': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'date_open': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'event_date': {
            'type': 'datetime',
            'format': '%Y-%m-%d',
        },
        'policy_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'claim_id': {
            'type': 'id',
            'subtype': 'string',
        },
        'customer_id': {
            'type': 'id',
            'subtype': 'string',
        },
    }

    metadata.add_table(
        name="customers",
        data=original_data["customers"],
        primary_key="customer_id",
        fields_metadata=customer_fields
    )

    metadata.add_table(
        name="policies",
        data=original_data["policies"],
        primary_key="policy_id",
        fields_metadata=policy_fields
    )

    metadata.add_table(
        name="claims",
        data=original_data["claims"],
        primary_key="claim_id",
        fields_metadata=claim_fields
    )

    metadata.add_relationship(
        parent="customers",
        child="policies",
        foreign_key="customer_id",
    )

    metadata.add_relationship(
        parent="policies",
        child="claims",
        foreign_key="policy_id",
    )

    metadata.add_relationship(
        parent="customers",
        child="claims",
        foreign_key="customer_id",
    )

    if save_metadata:
        save_metadata(metadata, dataset_name)
    return metadata 


