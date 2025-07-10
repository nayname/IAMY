def get_name(conn, type):
    """
    Provides information about the object type that needs validation.
    :param conn:
    :param lecture:
    :return:
    """
    return {"config_id": "", "title": "f[3]", "group_id": "f[4]"}


def report(lecture, version, conn):
    """
    Displays various accuracy metrics for the current preset, measured against the gold dataset and additional benchmarks.
    :param lecture:
    :param version:
    :param conn:
    :return:
    """
    name = get_name(conn, lecture)
    report = {}
    dataset_report = get_dataset_report(conn, lecture, version, [])

    report[name['group_id']] = {"name": name,
                       # "flow report": get_flow_report(conn, lecture, version),
                       "score": score(dataset_report),
                       "dataset report": dataset_report,
                       "version": version}
    return report


def score(report):
    """
    Custom score for a given version measured against the gold dataset.

    :param report:
    :param l:
    :param version:
    :param conn:
    :return:
    """
    return 0

def get_dataset_report(conn, lecture, version, not_valid_errors):
    """
    Confusion matrix
    :param conn:
    :param lecture:
    :param version:
    :param not_valid_errors:
    :return:
    """

    report = {"rows": {"all": "rowcount", "errors":0, "declined":0},"errors": {}}

    return report


def save_report(rprt, conn):
    """
    Store the results in the final report (final_predictions)
    :param rprt:
    :param conn:
    :return:
    """
    cur = conn.cursor()

    for r in rprt.keys():
        insert_query = ("INSERT INTO report.final_predictions(graded, overall, "
                        "percentages, group_id) VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT (group_id) DO UPDATE SET"
                        " percentages=%s, graded=%s, overall=%s")

        cur.execute(insert_query,
                    (rprt[r]['overall']['graded'], rprt[r]['overall']['overall'],
                     rprt[r]['overall']['percentages'], r, rprt[r]['overall']['percentages'],
                     rprt[r]['overall']['graded'], rprt[r]['overall']['overall']))
