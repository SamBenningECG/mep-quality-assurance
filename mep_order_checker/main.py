from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from typing import TypedDict
from dotenv import load_dotenv
from jinja2 import Template
import pymongo
import datetime
from mirakl_lib import (
    MiraklClient,
    MiraklOrder,
)
import pandas as pd  # type: ignore
from walmart_seller_lib import GetOrdersQueryParams, WalmartSellerClient  # type: ignore

#########################################################################
# Setting Up ENV Variables #
#########################################################################

load_dotenv()

KOHLS_API_KEY = os.getenv("KOHLS_API_KEY")
KOHLS_API_URL = os.getenv("KOHLS_BASE_URL")

KROGER_API_KEY = os.getenv("KROGER_API_KEY")
KROGER_API_URL = os.getenv("KROGER_BASE_URL")

OTC_API_KEY = os.getenv("OTC_API_KEY")
OTC_API_URL = os.getenv("OTC_BASE_URL")

SALONCENTRIC_API_KEY = os.getenv("SALONCENTRIC_API_KEY")
SALONCENTRIC_API_URL = os.getenv("SALONCENTRIC_BASE_URL")

GE_API_KEY = os.getenv("GE_API_KEY")
GE_API_URL = os.getenv("GE_BASE_URL")

WALMART_BASE_URL = os.getenv("WALMART_BASE_URL")
WALMART_CLIENT_ID = os.getenv("WALMART_CLIENT_ID")
WALMART_CLIENT_SECRET = os.getenv("WALMART_CLIENT_SECRET")


DB_NAME = "unfi"
DB_PORT = 25060
DB_USER = "doadmin"
DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MONGO_CONNECTION_STRING = "mongodb://localhost:27020"
MONGO_DB_NAME = "mirakl"

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
# EMAIL_RECIPIENTS = "dee@suburbanstreettrading.com,sam.b@elitecommercegroup.com"
# EMAIL_RECIPIENTS = "dee@suburbanstreettrading.com,sam.b@elitecommercegroup.com,eric@suburbanstreettrading.com,bob@suburbanstreettrading.com"
EMAIL_RECIPIENTS = "sam.b@elitecommercegroup.com"


UNFI_CLIENT_URL = os.getenv("UNFI_CLIENT_URL")

MONGO_CONNECTION_STRING = "mongodb://localhost:27020"
MONGO_DB_NAME = "mirakl"

#########################################################################
# Email Sending #
#########################################################################


def send_email(subject: str, body: str) -> None:

    if type(SENDER_EMAIL) is not str:
        raise ValueError("SENDER_EMAIL must be a string")
    if type(SENDER_PASSWORD) is not str:
        raise ValueError("SENDER_PASSWORD must be a string")
    if type(SMTP_SERVER) is not str:
        raise ValueError("SMTP_SERVER must be a string")

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = EMAIL_RECIPIENTS
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)


def send_email_with_csv(subject: str, body: str, csv_df: pd.DataFrame) -> None:

    if type(SENDER_EMAIL) is not str:
        raise ValueError("SENDER_EMAIL must be a string")
    if type(SENDER_PASSWORD) is not str:
        raise ValueError("SENDER_PASSWORD must be a string")
    if type(SMTP_SERVER) is not str:
        raise ValueError("SMTP_SERVER must be a string")

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = EMAIL_RECIPIENTS
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    csv = csv_df.to_csv(index=False)
    attachment = MIMEText(csv)
    attachment.add_header("Content-Disposition", "attachment", filename="orders.csv")
    msg.attach(attachment)

    with smtplib.SMTP(SMTP_SERVER, 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)


class TrackingEmailData(TypedDict):
    subject: str
    greeting: str
    message: str
    sender_name: str
    order_links: list[dict]


def send_test_email_with_html(
    data: TrackingEmailData, csv_df: pd.DataFrame | None = None
) -> None:

    if type(SENDER_EMAIL) is not str:
        raise ValueError("SENDER_EMAIL must be a string")
    if type(SENDER_PASSWORD) is not str:
        raise ValueError("SENDER_PASSWORD must be a string")
    if type(SMTP_SERVER) is not str:
        raise ValueError("SMTP_SERVER must be a string")

    # Read the Jinja2 email template
    with open("template.j2", "r") as file:
        template_str = file.read()

    jinja_template = Template(template_str)

    with smtplib.SMTP(SMTP_SERVER, 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = "sam.b@elitecommercegroup.com"
        msg["Subject"] = data["subject"]

        if csv_df is not None:
            csv = csv_df.to_csv(index=False)
            attachment = MIMEText(csv)
            attachment.add_header(
                "Content-Disposition", "attachment", filename="orders.csv"
            )
            msg.attach(attachment)

        email_content = jinja_template.render(data)

        msg.attach(MIMEText(email_content, "html"))

        server.send_message(msg)


#########################################################################
# Data Models #
#########################################################################


class OrderRecord(TypedDict):
    marketplace: str
    order_id: str
    order_status: str
    order_date: str
    shipping_deadline: str


#########################################################################
# Business Logic Utils #
#########################################################################


def add_business_days(start_date: datetime.date, days: int) -> datetime.date:
    current_date = start_date
    while days > 0:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() < 5:
            days -= 1
    return current_date


def determine_shipping_deadline(order: dict) -> str:

    match (order["marketplace"]):
        case "KROGER", "SOLONCENTRIC":
            shipping_deadline = datetime.datetime.fromisoformat(
                order["created_date"]
            ) + datetime.timedelta(days=5)
            return shipping_deadline.isoformat()
        case "OTC":
            created_date = datetime.datetime.fromisoformat(order["created_date"])
            return add_business_days(created_date, 3).isoformat()
        case _:  # default case
            raise ValueError("Marketplace not supported")


#########################################################################
# Mapper Functions #
#########################################################################


def mirakl_order_to_record(order: dict) -> OrderRecord:

    created_date = datetime.datetime.fromisoformat(order["created_date"])
    shipping_deadline = add_business_days(created_date, 3)

    return {
        "marketplace": order["marketplace"],
        "order_id": order["order_id"],
        "order_status": order["order_state"],
        "order_date": order["created_date"],
        "shipping_deadline": shipping_deadline.isoformat(),
    }


def walmart_order_to_record(order: dict) -> OrderRecord:

    created_date = datetime.datetime.fromtimestamp(order["order_date"] / 1000)
    shipping_deadline = add_business_days(created_date, 3)

    return {
        "marketplace": "WALMART",
        "order_id": order["purchase_order_id"],
        "order_status": order["order_lines"]["order_line"][0]["orderLineStatuses"][
            "orderLineStatus"
        ][0]["status"],
        "order_date": created_date.isoformat(),
        "shipping_deadline": shipping_deadline.isoformat(),
    }


#########################################################################
# MongoDb Data Access
#########################################################################


class MiraklOrderRepository:

    def __init__(self, connection_string, database_name):
        self.connection_string = connection_string
        self.database_name = database_name

    def __enter__(self):
        self.client = pymongo.MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        self.collection = self.db["miraklOrders"]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def upsert_order(self, order: MiraklOrder):
        self.collection.update_one(
            {"order_id": order.order_id}, {"$set": order.model_dump()}, upsert=True
        )

    def get_all_orders(self) -> list[dict]:
        return list(self.collection.find())

    def find_orders_after_date(self, date: str, order_status: str) -> list[dict]:
        return list(
            self.collection.find(
                {"created_date": {"$gt": date}, "order_state": order_status}
            )
        )


class WalmartOrderRepository:

    def __init__(self, connection_string, database_name):
        self.connection_string = connection_string
        self.database_name = database_name

    def __enter__(self):
        self.client = pymongo.MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        self.collection = self.db["walmartOrders"]

    def __exit__(self, exec_type, exc_val, exc_tb):
        self.client.close()

    def upsert_order(self, order: dict):
        self.collection.update_one(
            {"order_id": order["purchase_order_id"]}, {"$set": order}, upsert=True
        )

    def find_orders_after_date(self, date: str, order_status: str) -> list[dict]:

        date_epoch = (
            datetime.datetime.fromisoformat(date).timestamp() * 1000
        )  # convert to milliseconds

        # {"order_lines.order_line": {$elemMatch: {"orderLineStatuses.orderLineStatus": {$elemMatch: {"status": "Acknowledged"}}}}}

        return list(
            self.collection.find(
                {
                    "order_date": {"$gt": int(date_epoch)},
                    "order_lines.order_line.orderLineStatuses.orderLineStatus.status": order_status,
                }
            )
        )


#########################################################################
# Main #
#########################################################################


def sync_mirakl_orders(
    client: MiraklClient, repo: MiraklOrderRepository, start_date: str, end_date: str
) -> None:
    orders = client.fetch_orders(
        filter_params={"paginate": True, "start_date": start_date, "end_date": end_date}
    )

    for order in orders:
        order.marketplace = client.marketplace
        repo.upsert_order(order)


def sync_monthly_mirakl_orders(
    client: MiraklClient, repo: MiraklOrderRepository
) -> None:
    today = datetime.date.today()
    iso_date = today.isoformat()

    one_month_ago = datetime.date.today() - datetime.timedelta(days=30)
    one_month_ago_iso = one_month_ago.isoformat()

    sync_mirakl_orders(client, repo, one_month_ago_iso, iso_date)


def sync_monthly_walmart_orders(
    client: WalmartSellerClient, repo: WalmartOrderRepository
):
    today = datetime.date.today()
    iso_date = today.isoformat()

    one_month_ago = datetime.date.today() - datetime.timedelta(days=30)
    one_month_ago_iso = one_month_ago.isoformat()

    params = GetOrdersQueryParams(
        created_end_date=iso_date, created_start_date=one_month_ago_iso, limit="1000"
    )

    orders = client.get_orders(params=params).response_data.elements.order

    for order in orders:
        repo.upsert_order(order.model_dump())


def find_orders_needing_processing(repo: MiraklOrderRepository) -> list[dict]:

    today = datetime.date.today() - datetime.timedelta(days=1)
    today_iso = today.isoformat()

    orders = repo.find_orders_after_date(today_iso, "WAITING_ACCEPTANCE")

    return orders


def find_orders_needing_tracking(
    mirakl_repo: MiraklOrderRepository,
    walmart_repo: WalmartOrderRepository | None = None,
) -> list[OrderRecord]:

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    yesterday_iso = yesterday.isoformat()

    all_orders = []

    mirakl_orders = mirakl_repo.find_orders_after_date(yesterday_iso, "SHIPPING")

    all_orders.extend([mirakl_order_to_record(order) for order in mirakl_orders])

    if walmart_repo is not None:
        walmart_orders = walmart_repo.find_orders_after_date(
            yesterday_iso, "Acknowledged"
        )
        all_orders.extend([walmart_order_to_record(order) for order in walmart_orders])

    return all_orders


def get_mirakl_orders_need_placement(repo: MiraklOrderRepository) -> list[dict]:
    raise NotImplementedError


def generate_order_link(order: OrderRecord) -> str:

    if order["marketplace"] == "WALMART":
        return f"https://seller.walmart.com/orders/manage-orders?orderGroups=Unshipped&poNumber={order['order_id']}"
    elif order["marketplace"] == "KROGER":
        return f"https://kroger-prod.mirakl.net/mmp/shop/order/{order['order_id']}"
    elif order["marketplace"] == "OTC":
        return f"https://orientaltradingus-prod.mirakl.net/mmp/shop/order/{order['order_id']}"
    elif order["marketplace"] == "SALONCENTRIC":
        return (
            f"https://saloncentricus-prod.mirakl.net/mmp/shop/order/{order['order_id']}"
        )
    else:
        return "https://www.google.com"


def prepare_tracking_email_data(orders: list[OrderRecord]) -> TrackingEmailData:

    data: TrackingEmailData = {
        "subject": "Tracking Update Summary",
        "greeting": "Team,",
        "message": "The following orders need tracking updates.",
        "sender_name": "MEP Automation System",
        "order_links": [],
    }

    for order in orders:

        order_link = generate_order_link(order)
        data["order_links"].append({"text": order["order_id"], "link": order_link})

    return data


def main():

    if (
        WALMART_BASE_URL is None
        or WALMART_CLIENT_ID is None
        or WALMART_CLIENT_SECRET is None
    ):
        raise ValueError("Walmart ENV variables not set")

    walmart_client = WalmartSellerClient(
        base_url=WALMART_BASE_URL,
        client_id=WALMART_CLIENT_ID,
        client_secret=WALMART_CLIENT_SECRET,
    )
    walmart_repo = WalmartOrderRepository(MONGO_CONNECTION_STRING, MONGO_DB_NAME)

    kohls_client = MiraklClient("KOHLS", KOHLS_API_URL, KOHLS_API_KEY)
    kroger_client = MiraklClient("KROGER", KROGER_API_URL, KROGER_API_KEY)
    otc_client = MiraklClient("OTC", OTC_API_URL, OTC_API_KEY)
    saloncentric_client = MiraklClient(
        "SALONCENTRIC", SALONCENTRIC_API_URL, SALONCENTRIC_API_KEY
    )
    ge_client = MiraklClient("GE", GE_API_URL, GE_API_KEY)

    mirakl_clients = [
        kohls_client,
        kroger_client,
        otc_client,
        saloncentric_client,
        ge_client,
    ]

    mirakl_repo = MiraklOrderRepository(MONGO_CONNECTION_STRING, MONGO_DB_NAME)

    with mirakl_repo, walmart_repo:

        sync_monthly_walmart_orders(walmart_client, walmart_repo)

        for client in mirakl_clients:
            sync_monthly_mirakl_orders(client, mirakl_repo)
        orders_needing_processing = find_orders_needing_processing(mirakl_repo)
        orders_needing_tracking = find_orders_needing_tracking(
            mirakl_repo, walmart_repo
        )

        processing_df = pd.DataFrame(orders_needing_processing)
        tracking_df = pd.DataFrame(orders_needing_tracking)

        print(tracking_df)

        if len(orders_needing_processing) > 0:
            send_email(
                "Order Processing Summary",
                "The Following orders are more than 1 day old and need processing",
            )

            send_email_with_csv(
                "Order Processing Summary",
                "The Following orders are more than 1 day old and need processing",
                processing_df,
            )

        else:
            send_email(
                "Order Processing Summary",
                "There are no orders older than 1 day which need processing",
            )

        if len(orders_needing_tracking) > 0:
            send_test_email_with_html(
                prepare_tracking_email_data(orders_needing_tracking), tracking_df
            )

        else:
            send_email(
                "Tracking Update Summary",
                "There are no orders older than 1 day which need tracking",
            )


def test():

    if (
        WALMART_BASE_URL is None
        or WALMART_CLIENT_ID is None
        or WALMART_CLIENT_SECRET is None
    ):
        raise ValueError("Walmart ENV variables not set")

    walmart_client = WalmartSellerClient(
        base_url=WALMART_BASE_URL,
        client_id=WALMART_CLIENT_ID,
        client_secret=WALMART_CLIENT_SECRET,
    )
    walmart_repo = WalmartOrderRepository(MONGO_CONNECTION_STRING, MONGO_DB_NAME)

    with walmart_repo:
        sync_monthly_walmart_orders(walmart_client, walmart_repo)


if __name__ == "__main__":
    main()
    # send_test_email_with_html()
