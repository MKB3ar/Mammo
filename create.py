from kafka.admin import KafkaAdminClient, NewTopic

admin_client = KafkaAdminClient(
    bootstrap_servers='localhost:9092',
    client_id='mammo-admin'
)

topic = NewTopic(name="mammo_results", num_partitions=1, replication_factor=1)

try:
    admin_client.create_topics(new_topics=[topic], validate_only=False)
    print("✅ Топик mammo_results создан")
except Exception as e:
    print("❌ Ошибка при создании топика:", str(e))