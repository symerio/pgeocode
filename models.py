# A list of the util functions


def is_valid_zip_brazil(zip_code):
    if isinstance(zip_code, list):
        zip_code = ''.join(zip_code)
    if zip_code[0].find('-'):
        zip_code = ''.join(zip_code.split('-'))

    result = []

    result.append(1) if zip_code[5:] == '000' else result.append(0)

    result.append(1) if len(zip_code) == 8 else result.append(0)

    return all(result)


def check_is_valid_zip_code(zip_code):
    if isinstance(zip_code, list):
        zip_code = ''.join(zip_code)

    msg = f'zip_code {zip_code} is invalid for Brazil'
    if not is_valid_zip_brazil(zip_code):
        return msg
