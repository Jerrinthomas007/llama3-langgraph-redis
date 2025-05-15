import dns.message
import dns.query
import dns.rdatatype

def get_weather(city: str) -> str:
    domain = f"{city.lower()}.weather"
    query = dns.message.make_query(domain, dns.rdatatype.TXT)

    try:
        # Send over UDP to dns.toys directly (correct IP)
        response = dns.query.udp(query, '204.48.19.68', timeout=3)

        if not response.answer:
            return "Error: No weather info returned."

        weather = []
        for answer in response.answer:
            for item in answer.items:
                weather.append(item.to_text().strip('"'))

        return "\n".join(weather)
    except Exception as e:
        return f"Error: {e}"
