"""
Parser for sample lists from Excel files.
"""

import pandas as pd
import logging
from battery_analysis import models


def parse_sample_list(file_path):
    """
    Parse an Excel file containing a list of battery samples.

    Flexible column naming:
    - Name/Sample/ID/Cell ID: Sample name (required)
    - Chemistry/Material/Composition: Cell chemistry
    - Form Factor/Type/Format: Cell form factor
    - Manufacturer/Vendor/Source: Cell manufacturer
    - Capacity/Nominal Capacity/Rated Capacity: Rated capacity (mAh)

    Args:
        file_path: Path to the Excel file

    Returns:
        tuple: (parsed_samples, error_message)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Show available columns for debugging
        logging.info(f"Columns in Excel file: {list(df.columns)}")

        # Find name column - try various possible names
        name_column = None
        name_options = [
            "Name",
            "Sample",
            "Sample Name",
            "ID",
            "Cell ID",
            "Cell Name",
            "Identifier",
        ]
        for option in name_options:
            if option in df.columns:
                name_column = option
                break

        if not name_column:
            return (
                [],
                f"Excel file must contain a name column. Available columns: {', '.join(df.columns)}",
            )

        # Map other common column names
        column_mappings = {
            "chemistry": ["Chemistry", "Material", "Composition", "Cell Chemistry"],
            "form_factor": ["Form Factor", "Type", "Format", "Form", "Cell Type"],
            "manufacturer": ["Manufacturer", "Vendor", "Source", "Company", "Provider"],
            "nominal_capacity": [
                "Capacity",
                "Nominal Capacity",
                "Rated Capacity",
                "Capacity (mAh)",
                "Nominal Capacity (mAh)",
            ],
        }

        # Find actual column names
        field_columns = {}
        for field, options in column_mappings.items():
            for option in options:
                if option in df.columns:
                    field_columns[field] = option
                    break

        # Process each row
        samples = []
        for _, row in df.iterrows():
            # Skip rows with empty name
            if pd.isna(row[name_column]) or row[name_column] == "":
                continue

            sample_data = {
                "name": str(row[name_column]),
            }

            # Add optional fields if present
            for field, column in field_columns.items():
                if not pd.isna(row[column]):
                    if field == "nominal_capacity":
                        # Convert to float for capacity
                        try:
                            sample_data[field] = float(row[column])
                        except ValueError:
                            # Skip invalid values
                            pass
                    else:
                        sample_data[field] = str(row[column])

            samples.append(sample_data)

        return samples, None

    except Exception as e:
        logging.exception("Error parsing sample list")
        return [], f"Error parsing sample list: {str(e)}"


def import_samples(file_path, update_existing=True):
    """
    Import samples from an Excel file into the database.

    Args:
        file_path: Path to the Excel file
        update_existing: Whether to update existing samples

    Returns:
        tuple: (success_count, updated_count, error_message)
    """
    # Parse the sample list
    samples, error = parse_sample_list(file_path)
    if error:
        return 0, 0, error

    success_count = 0
    updated_count = 0

    # Import each sample
    for sample_data in samples:
        try:
            # Check if the sample already exists
            existing = models.Sample.get_by_name(sample_data["name"])

            if existing and update_existing:
                # Update existing sample
                for key, value in sample_data.items():
                    if key != "name":  # Don't update name
                        setattr(existing, key, value)

                existing.save()
                updated_count += 1
            elif not existing:
                # Create new sample
                sample = models.Sample(**sample_data)
                sample.save()
                success_count += 1

        except Exception as e:
            # Log error but continue processing
            logging.error(
                f"Error importing sample {sample_data.get('name', 'unknown')}: {str(e)}"
            )

    return success_count, updated_count, None
