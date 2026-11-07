"""Write the organiser's exact CSV format so regression can compare bytes."""

import os


def write_submission(path, labels):
    """Preserve the starter's header, frame numbering, and newline handling."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path
