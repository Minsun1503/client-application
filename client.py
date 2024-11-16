import psutil
import GPUtil
import cpuinfo
import time
import numpy as np
import math
import multiprocessing
import psycopg2
from psycopg2 import sql
import platform
import subprocess
import requests
import redis
from time import sleep
try:
    import cupy as cp
except ImportError:
    import pyopencl as cl

quit = False


def get_alternative_gpu_info():
    """Lấy thông tin GPU bằng cách sử dụng PowerShell hoặc lệnh hệ thống."""
    try:
        if platform.system() == "Windows":
            # Sử dụng PowerShell để lấy thông tin GPU
            result = subprocess.run(['powershell', 'Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion'], 
                                    stdout=subprocess.PIPE, text=True)
            return result.stdout.strip()
        elif platform.system() == "Linux":
            # Sử dụng lệnh lspci cho Linux
            result = subprocess.run(['lspci', '-vnn'], stdout=subprocess.PIPE, text=True)
            return result.stdout.strip()
    except Exception as e:
        return f"Error retrieving GPU info with alternative method: {e}"
    return None


def parse_alternative_gpu_info(raw_info):
    """Phân tích kết quả từ PowerShell hoặc lệnh hệ thống và định dạng lại cho dễ đọc."""
    lines = raw_info.split('\n')
    parsed_info = {}

    for line in lines[1:]:  # Bỏ qua dòng tiêu đề
        parts = line.split()
        if len(parts) >= 3:
            parsed_info = {
                "GPU Name": " ".join(parts[:-2]),
                "Adapter RAM": parts[-2],
                "Driver Version": parts[-1]
            }
    gpu_contribution = int(input("Bạn muốn chúng tôi dùng bao nhiêu dung lượng GPU khả thi của bạn (1-100)%: "))
    if not (1 <= gpu_contribution <= 100):
        print("Con số bạn nhập không khả dụng, vui lòng nhập từ 1 đến 100.")
        return
    parsed_info["GPU Contribute"] = gpu_contribution
    print("Bắt đầu benchmark GPU...")
    gpu_score = benchmark_gpu()
    parsed_info["GPU Score"] = gpu_score
    return parsed_info


def get_full_system_info():
    global quit
    print("Chào mừng bạn đến với Minnsun'Bow, để bắt đầu vui lòng đăng nhập hoặc tạo tài khoản.")
    print("Nếu chưa có tài khoản vui lòng tạo tài khoản bằng cách gõ 'register'")
    print("Nếu đã có tài khoản vui lòng đăng nhập bằng cách gõ 'login'")
    print("Hoặc gõ 'no' để thoát chương trình")
    get_register = input()
    if get_register.lower() == "register":
        # Lấy tên của client
        client_name = input("Vui lòng nhập tài khoản của bạn: ").strip()
        if not client_name:
            print("Tài khoản không được để trống!")
            exit()
        password = input("Vui lòng nhập mật khẩu của bạn: ").strip()
        if not password:
            print("Passwork không được để trống!")
            exit()
        client_info = {
            "client_name": client_name,
            "password": password
        }
        get_infor = input("Chúng tôi sẽ bắt đầu kiểm tra hệ thống để chẩn đoán tình trạng CPU, GPU của bạn? Chọn 'yes' để bắt đầu hoặc 'no' để từ chối: ")
        if get_infor.lower() != "yes":
            print("Bạn đã chọn 'no', chương trình sẽ kết thúc. Cảm ơn bạn!")
            quit = True
            return
        # Lấy thông tin CPU
        try:
            cpu = cpuinfo.get_cpu_info()
            if cpu:
                cpu_contribution = int(input("Bạn muốn chúng tôi dùng bao nhiêu dung lượng CPU khả thi của bạn (1-100)%: "))
                if not (1 <= cpu_contribution <= 100):
                    print("Con số bạn nhập không khả dụng, vui lòng nhập từ 1 đến 100.")
                    return
                print("Bắt đầu benchmark CPU...")
                # Benchmark CPU
                cpu_score = benchmark_cpu()

                # Thông tin cơ bản về CPU
                cpu_info = {
                    "client_name": client_name,
                    "CPU Name": cpu.get("brand_raw", "Unknown CPU"),
                    "CPU Architecture": cpu.get("arch", "Unknown"),
                    "CPU Bits": cpu.get("bits", 64),
                    "Advertised Frequency": cpu.get("hz_advertised_friendly", "N/A"),
                    "Actual Frequency": cpu.get("hz_actual_friendly", "N/A"),
                    "L2 Cache": cpu.get("l2_cache_size", "N/A"),
                    "L3 Cache": cpu.get("l3_cache_size", "N/A"),
                    "CPU Count (Logical)": psutil.cpu_count(logical=True),
                    "CPU Count (Physical)": psutil.cpu_count(logical=False),
                    "Total Memory (MB)": psutil.virtual_memory().total // (1024 ** 2),
                    "Available Memory (MB)": psutil.virtual_memory().available // (1024 ** 2),
                    "CPU Usage (%)": psutil.cpu_percent(interval=1),
                    "Detailed Info": cpu.get("flags", []),
                    "CPU Contribute": cpu_contribution,
                    "CPU Score": cpu_score
                }
            else:
                print("Không tìm thấy thông tin CPU.")
                cpu_info, cpu_score = None, 0
        except Exception as e:
            print(f"Error retrieving CPU info: {e}")
            cpu_info, cpu_score = None, 0

        # Lấy thông tin GPU
        gpu_info_list = []
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_contribution = int(input("Bạn muốn chúng tôi dùng bao nhiêu dung lượng GPU khả thi của bạn (1-100)%: "))
                if not (1 <= gpu_contribution <= 100):
                    print("Con số bạn nhập không khả dụng, vui lòng nhập từ 1 đến 100.")
                    return
                print("Bắt đầu benchmark GPU...")
                # Benchmark GPU
                gpu_score = benchmark_gpu()

                for gpu in gpus:
                    gpu_info = {
                        "GPU Name": gpu.name,
                        "GPU Load (%)": gpu.load * 100,
                        "GPU Memory Total (MB)": gpu.memoryTotal,
                        "GPU Memory Free (MB)": gpu.memoryFree,
                        "GPU Temperature (°C)": gpu.temperature,
                        "GPU Contribute": gpu_contribution,
                        "GPU Score": gpu_score
                    }
                    gpu_info_list.append(gpu_info)
            else:
                print("Không tìm thấy GPU NVIDIA. Sử dụng phương pháp thay thế...")
                print("Đang tiến hành xử lí...")
                alternative_info = get_alternative_gpu_info()
                if alternative_info:
                    parsed_gpu_info = parse_alternative_gpu_info(alternative_info)
                    gpu_info_list.append(parsed_gpu_info)
        except Exception as e:
            print(f"Error retrieving GPU info: {e}")
            gpu_info_list = None

        # In thông tin CPU và GPU
        if cpu_info:
            print("\n--- CPU Information ---")
            for key, value in cpu_info.items():
                print(f"{key}: {value}")
        else:
            print("\nNo CPU information available.")

        if gpu_info_list:
            print("\n--- GPU Information ---")
            for gpu_info in gpu_info_list:
                for key, value in gpu_info.items():
                    print(f"{key}: {value}")
        else:
            print("\nNo GPU information available.")

        # Gửi thông tin về database
        send_resources(client_info, cpu_info, gpu_info_list)
    elif get_register.lower() == "login":
        get_login()
    elif get_register.lower() == "no":
        print("Bạn đã chọn 'no' để không thực hiện bất kỳ hành động nào. Chào tạm biệt!")
        exit()
    else:
        print("Bạn đang nhập không đúng, vui lòng thử lại!")


# Kết nối đến Redis
def connect_to_redis():
    try:
        redis_client = redis.StrictRedis(
            host="singapore-redis.render.com",  # Đảm bảo sử dụng đúng host
            port=6379,
            username="red-csqpikaj1k6c73c0lu20",
            password="R4hYq4YYA56dMDn2a0cSw2B7rxIIivgo",
            decode_responses=True,
            ssl=True,  # Sử dụng SSL khi kết nối Redis trên Render
            socket_timeout=10
        )
        redis_client.ping()  # Kiểm tra kết nối
        print("Kết nối Redis thành công.")
        return redis_client
    except redis.exceptions.ConnectionError as e:
        print(f"Có lỗi khi kết nối Redis: {e}")
        return None


# Tạo kết nối Redis ban đầu
redis_client = connect_to_redis()


# Hàm đăng nhập và kết nối tới Redis
def get_login():
    client_name = input("Nhập tên đăng nhập: ")
    password = input("Nhập mật khẩu: ")
    print("Đang tiến hành kết nối đến máy chủ...")

    try:
        # Kết nối tới cơ sở dữ liệu
        conn = psycopg2.connect(
            dbname="minnsundb",
            user="minnsundb_owner",
            password="Qad5JKh3cYsD",
            host="ep-polished-forest-a14xdih0.ap-southeast-1.aws.neon.tech",
            port="5432"
        )
        cursor = conn.cursor()
        # Truy vấn để kiểm tra tài khoản và mật khẩu
        query = "SELECT * FROM client_info WHERE client_name = %s AND password = %s"
        cursor.execute(query, (client_name, password))

        # Lấy kết quả
        result = cursor.fetchone()

        if result:
            print("Đăng nhập thành công!")
            cpu_status = "green"  # Ví dụ, có thể thay bằng trạng thái thực tế của CPU
            gpu_status = "green"  # Ví dụ, có thể thay bằng trạng thái thực tế của GPU
            send_status_to_server(client_name, cpu_status, gpu_status)
            maintain_connection(client_name)  # Duy trì kết nối và lắng nghe nhiệm vụ
        else:
            print("Tài khoản hoặc mật khẩu không đúng.")

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Gửi trạng thái kết nối của client tới Redis
def send_status_to_server(client_name, cpu_status, gpu_status):
    try:
        redis_client.hset("devices_status", client_name, f"CPU: {cpu_status}, GPU: {gpu_status}")
        print(f"Trạng thái đã được cập nhật cho client {client_name}")
    except Exception as e:
        print(f"Có lỗi khi gửi trạng thái: {e}")


# Hàm duy trì kết nối và chờ nhiệm vụ
def maintain_connection(client_name):
    global redis_client
    print(f"{client_name} đang duy trì kết nối và chờ nhiệm vụ...")
    while True:
        if redis_client is None:
            print("Kết nối Redis đã bị đóng. Thử kết nối lại...")
            redis_client = connect_to_redis()
            if redis_client is None:
                time.sleep(5)  # Nếu không thể kết nối lại ngay, thử lại sau 5 giây
                continue

        # Chờ nhiệm vụ mới từ Redis (giả sử nhiệm vụ được lưu trong một queue)
        task = redis_client.lpop("task_queue")  # Lấy nhiệm vụ từ Redis queue
        if task:
            print(f"Nhận nhiệm vụ mới: {task}")
            # Xử lý nhiệm vụ (ví dụ: tính toán, render, v.v.)
        else:
            time.sleep(1)  # Nếu không có nhiệm vụ, chờ 1 giây trước khi kiểm tra lại


def benchmark_gpu():
    """
    Thực hiện benchmark GPU bằng cách xử lý một ma trận lớn cho tất cả các loại GPU hiện có.
    Nếu GPU hỗ trợ CUDA (NVIDIA), sử dụng `cupy`. Nếu không, sử dụng OpenCL cho các GPU khác.
    """
    # Kích thước ma trận cho bài kiểm tra
    matrix_size = 4096  # Kích thước ma trận (4096x4096)
    np.random.seed(42)  # Đặt hạt giống để tạo ra một ma trận cố định cho tất cả các lần chạy
    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    try:
        # Kiểm tra nếu có GPU NVIDIA với CUDA
        if 'cupy' in globals():
            # Sử dụng GPU NVIDIA với CUDA thông qua CuPy
            a_gpu = cp.array(a)
            b_gpu = cp.array(b)
            start_time = time.time()
            c_gpu = cp.matmul(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()  # Đảm bảo hoàn thành tính toán
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Đã sử dụng GPU NVIDIA với CUDA")
        else:
            print("Không tìm thấy CuPy, chuyển sang OpenCL cho GPU khác.")
            print("Đang xử lí...")
            # Nếu không có CuPy, thử OpenCL cho các GPU không phải NVIDIA
            platforms = cl.get_platforms()
            devices = platforms[0].get_devices(device_type=cl.device_type.GPU)  # Lựa chọn GPU đầu tiên

            if not devices:
                print("Không tìm thấy GPU hỗ trợ OpenCL.")
                return 0

            device = devices[0]
            context = cl.Context([device])
            queue = cl.CommandQueue(context)

            # Tạo bộ nhớ trên GPU cho ma trận a, b và c
            mf = cl.mem_flags
            a_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            b_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
            c_gpu = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)

            # Viết mã nguồn OpenCL để thực hiện phép nhân ma trận
            kernel_code = """
            __kernel void matmul(__global const float* a, __global const float* b, __global float* c, int n) {
                int row = get_global_id(0);
                int col = get_global_id(1);
                float sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    sum += a[row * n + i] * b[i * n + col];
                }
                c[row * n + col] = sum;
            }
            """

            # Biên dịch mã nguồn OpenCL và tạo kernel
            program = cl.Program(context, kernel_code).build()
            kernel = program.matmul

            # Thực thi benchmark
            start_time = time.time()
            kernel.set_arg(0, a_gpu)
            kernel.set_arg(1, b_gpu)
            kernel.set_arg(2, c_gpu)
            kernel.set_arg(3, np.int32(matrix_size))
            cl.enqueue_nd_range_kernel(queue, kernel, (matrix_size, matrix_size), None)
            queue.finish()  # Chờ đến khi tính toán hoàn tất
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Đã sử dụng GPU với OpenCL")

    except Exception as e:
        print(f"Lỗi khi sử dụng OpenCL hoặc CUDA: {e}")
        return 0

    # Tính toán điểm GPU dựa trên thời gian thực hiện
    base_score = 1000000  # Điểm chuẩn
    gpu_score = max(1, int(base_score / elapsed_time))  # Tính điểm

    print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")
    print(f"Điểm hiệu suất GPU: {gpu_score} (Dựa trên thời gian thực hiện)")

    return gpu_score


def heavy_computation(num_operations):
    """
    Hàm tính toán phức tạp.
    Sử dụng một bài toán cố định để đảm bảo chuẩn xác khi so sánh giữa các client.
    """
    result = 0
    for i in range(num_operations):
        result += math.sqrt(i)
    return result


def benchmark_cpu():
    """
    Thực hiện benchmark CPU bằng cách tính toán một số lượng lớn phép toán.
    Sử dụng tất cả các lõi CPU để đánh giá hiệu suất và trả về điểm CPU.
    """
    # Chỉ sử dụng một bài toán có số phép toán cố định
    num_operations = 10**7  # Giới hạn số phép toán, có thể điều chỉnh
    start_time = time.time()

    # Sử dụng tất cả các lõi CPU
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(heavy_computation, [num_operations] * multiprocessing.cpu_count())

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Tính toán điểm hiệu suất CPU
    base_score = 1000000  # Điểm chuẩn
    cpu_score = max(1, int(base_score / elapsed_time))  # Tính điểm

    print(f"Thời gian thực hiện: {elapsed_time:.2f} giây")
    print(f"Điểm hiệu suất CPU: {cpu_score} (Dựa trên thời gian thực hiện)")

    return cpu_score


def send_resources(client_info, cpu_info, gpu_info_list):
    try:
        # Kết nối tới cơ sở dữ liệu
        conn = psycopg2.connect(
            dbname="minnsundb",
            user="minnsundb_owner",
            password="Qad5JKh3cYsD",
            host="ep-polished-forest-a14xdih0.ap-southeast-1.aws.neon.tech",
            port="5432"
        )
        cursor = conn.cursor()

        # Truy vấn để thêm hoặc cập nhật thông tin người dùng
        query_client = """
        INSERT INTO client_info (client_name, password)
        VALUES (%s, %s)
        ON CONFLICT (client_name) DO UPDATE
        SET password = EXCLUDED.password
        """

        cursor.execute(query_client, (
            client_info["client_name"],
            client_info["password"]
        ))
        # Truy vấn để thêm hoặc cập nhật thông tin CPU
        query_cpu = """
        INSERT INTO cpu_info (client_name, cpu_name, memory_total, memory_available, cpu_contribution, cpu_score)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (client_name) DO UPDATE
        SET cpu_name = EXCLUDED.cpu_name,
            memory_total = EXCLUDED.memory_total,
            memory_available = EXCLUDED.memory_available,
            cpu_contribution = EXCLUDED.cpu_contribution,
            cpu_score = EXCLUDED.cpu_score
        """
        cursor.execute(query_cpu, (
            cpu_info["client_name"],
            cpu_info["CPU Name"],
            cpu_info["Total Memory (MB)"],
            cpu_info["Available Memory (MB)"],
            cpu_info["CPU Contribute"],
            cpu_info["CPU Score"]
        ))

        # Nếu có thông tin GPU, thêm vào bảng gpu_info
        if gpu_info_list:
            query_gpu = """
            INSERT INTO gpu_info (client_name, gpu_name, gpu_contribution, gpu_score)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (client_name, gpu_name) DO UPDATE
            SET gpu_contribution = EXCLUDED.gpu_contribution,
                gpu_score = EXCLUDED.gpu_score
            """
            for gpu_info in gpu_info_list:
                cursor.execute(query_gpu, (
                    cpu_info["client_name"],
                    gpu_info["GPU Name"],
                    gpu_info["GPU Contribute"],
                    gpu_info["GPU Score"]
                ))

        # Commit giao dịch
        conn.commit()
        print("Dữ liệu đã được gửi thành công.")
        print("Đã đăng kí thành công.")

    except psycopg2.OperationalError as op_err:
        print(f"Lỗi kết nối cơ sở dữ liệu: {op_err}")
    except psycopg2.IntegrityError as int_err:
        print(f"Lỗi khi thực thi truy vấn: {int_err}")
    except psycopg2.Error as db_err:
        print(f"Lỗi cơ sở dữ liệu: {db_err}")
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

    finally:
        # Đảm bảo rằng kết nối và cursor được đóng dù có lỗi hay không
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except Exception as close_err:
            print(f"Lỗi khi đóng kết nối hoặc cursor: {close_err}")


if __name__ == "__main__":
    while not quit:
        get_full_system_info()
