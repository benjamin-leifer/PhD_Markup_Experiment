import shutil


class ScheduleStep:
    def __init__(self, label, ctrl_type, ctrl_value, limits):
        self.label = label
        self.ctrl_type = ctrl_type
        self.ctrl_value = ctrl_value
        self.limits = limits  # List of dictionaries, each representing a limit

    def to_sdu_format(self, step_index):
        step_section = f"[Schedule_Step{step_index}]"
        step_lines = [
            f"m_uLimitNum={len(self.limits)}",
            f"m_szLabel={self.label}",
            f"m_szStepCtrlType={self.ctrl_type}",
            f"m_szCtrlValue={self.ctrl_value}",
        ]
        limit_sections = {}
        for i, limit in enumerate(self.limits):
            limit_section = f"{step_section}_Limit{i}"
            limit_lines = [
                f"m_bStepLimit={limit['step_limit']}",
                f"m_bLogDataLimit={limit['log_data_limit']}",
                f"m_szGotoStep={limit['goto_step']}",
                f"Equation0_szLeft={limit['equation_left']}",
                f"Equation0_szCompareSign={limit['compare_sign']}",
                f"Equation0_szRight={limit['equation_right']}",
            ]
            limit_sections[limit_section] = limit_lines
        return step_section, step_lines, limit_sections


def read_sdu_file(file_path):
    sections = {}
    current_section = None

    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

    return sections


def write_sdu_file(file_path, sections):
    with open(file_path, 'w', encoding='ISO-8859-1') as file:
        for section, lines in sections.items():
            file.write(f"{section}\n")
            for line in lines:
                file.write(f"{line}\n")
            file.write("\n")


def copy_and_modify_sdu(template_path, output_path, schedule_steps):
    # Copy the template file to the output file
    shutil.copyfile(template_path, output_path)

    # Read the copied file
    sections = read_sdu_file(output_path)

    # Add new schedule steps
    step_index = len([section for section in sections if section.startswith('[Schedule_Step')])
    for step in schedule_steps:
        step_section, step_lines, limit_sections = step.to_sdu_format(step_index)
        sections[step_section] = step_lines
        sections.update(limit_sections)
        step_index += 1

    # Write the modified file
    write_sdu_file(output_path, sections)


# Example usage
template_path = 'NMC-Coin_1_Cycle_C10_Template_doc.sdu'  # Path to your template file
output_path = 'output_t1.sdu'  # Path to your output file

schedule_steps = [
    ScheduleStep(
        label="Step 1",
        ctrl_type="C-Rate",
        ctrl_value="0.1",
        limits=[
            {
                "step_limit": 1,
                "log_data_limit": 1,
                "goto_step": "Next Step",
                "equation_left": "PV_CHAN_Step_Time",
                "compare_sign": ">=",
                "equation_right": "600"
            }
        ]
    ),
    ScheduleStep(
        label="Step 2",
        ctrl_type="Rest",
        ctrl_value="0",
        limits=[
            {
                "step_limit": 1,
                "log_data_limit": 1,
                "goto_step": "Next Step",
                "equation_left": "DV_Time",
                "compare_sign": ">=",
                "equation_right": "60"
            }
        ]
    )
]

copy_and_modify_sdu(template_path, output_path, schedule_steps)
