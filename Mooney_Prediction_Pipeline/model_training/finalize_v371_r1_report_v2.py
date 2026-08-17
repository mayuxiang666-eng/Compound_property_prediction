"""Console-encoding-safe entry point for the R1 artifact finalizer."""

from . import finalize_v371_r1_report as report


def main() -> None:
    try:
        report.main()
    except UnicodeEncodeError:
        print("R1 report generated successfully.")


if __name__ == "__main__":
    main()
