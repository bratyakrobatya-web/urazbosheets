import streamlit as st
import pandas as pd
import anthropic
import io
from typing import Dict, Tuple
import time

st.set_page_config(page_title="Генератор заданий", layout="wide")

def parse_prompt_file(content: str) -> Dict[str, str]:
    """Парсит файл promt и извлекает инструкции для каждого типа задания."""
    prompts = {}
    lines = content.split('\n')

    current_type = None
    current_prompt = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Ищем строки с типами заданий (заканчиваются на табуляцию и начинаются с "Задания")
        if '\t' in lines[i] and line.startswith('Задания'):
            # Сохраняем предыдущий промпт
            if current_type and current_prompt:
                prompts[current_type] = '\n'.join(current_prompt)

            # Разделяем по табуляции
            parts = lines[i].split('\t')
            if len(parts) >= 2:
                current_type = parts[0].strip()
                current_prompt = [parts[1].strip()]
            i += 1
        elif current_type:
            # Добавляем строки к текущему промпту
            if line and not line.startswith('→'):
                current_prompt.append(line)
            i += 1
        else:
            i += 1

    # Сохраняем последний промпт
    if current_type and current_prompt:
        prompts[current_type] = '\n'.join(current_prompt)

    return prompts


def generate_task(client: anthropic.Anthropic,
                  prompt_template: str,
                  competence: str,
                  indicator: str,
                  discipline: str,
                  task_type: str) -> Tuple[str, str]:
    """Генерирует задание и ключ используя Claude API."""

    # Формируем полный промпт
    full_prompt = f"""
{prompt_template}

Данные для генерации:
- Компетенция (столбец A): {competence}
- Индикатор (столбец B): {indicator}
- Дисциплина/модуль/практика (столбец C): {discipline}
- Тип задания (столбец D): {task_type}

Верни результат СТРОГО в формате:
ЗАДАНИЕ:
[текст задания]

КЛЮЧ:
[текст ключа/ответа]

Важно: не добавляй никаких дополнительных комментариев, только задание и ключ в указанном формате.
"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        response_text = message.content[0].text

        # Парсим ответ
        if "ЗАДАНИЕ:" in response_text and "КЛЮЧ:" in response_text:
            parts = response_text.split("КЛЮЧ:")
            task = parts[0].replace("ЗАДАНИЕ:", "").strip()
            key = parts[1].strip()
            return task, key
        else:
            return "Ошибка генерации", "Ошибка генерации"

    except Exception as e:
        return f"Ошибка: {str(e)}", f"Ошибка: {str(e)}"


def main():
    st.title("🎓 Генератор учебных заданий")
    st.markdown("---")

    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        api_key = st.text_input("Claude API ключ:", type="password",
                                help="Введите ваш API ключ от Anthropic")

        st.markdown("---")
        st.markdown("### 📋 Инструкция")
        st.markdown("""
        1. Введите API ключ
        2. Загрузите файл Excel с данными
        3. Загрузите файл с промптами
        4. Выберите строки для обработки
        5. Нажмите "Сгенерировать задания"
        """)

    # Основная часть
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Загрузка Excel файла")
        excel_file = st.file_uploader(
            "Загрузите файл megaphops.xlsx или аналогичный",
            type=['xlsx', 'xls'],
            help="Файл должен содержать столбцы: Компетенция, Индикатор, Дисциплина, Уровень сложности, Задание, Ключ"
        )

    with col2:
        st.subheader("📝 Загрузка файла с промптами")
        prompt_file = st.file_uploader(
            "Загрузите файл promt",
            type=['txt'],
            help="Файл с инструкциями для генерации различных типов заданий"
        )

    if excel_file and prompt_file:
        # Читаем файлы
        df = pd.read_excel(excel_file)
        prompt_content = prompt_file.read().decode('utf-8')
        prompts = parse_prompt_file(prompt_content)

        st.success(f"✅ Загружено {len(df)} строк из Excel файла")
        st.info(f"📚 Найдено {len(prompts)} типов заданий в файле промптов")

        # Показываем превью данных
        with st.expander("👀 Просмотр данных (первые 5 строк)"):
            st.dataframe(df.head())

        st.markdown("---")

        # Настройки обработки
        st.subheader("⚙️ Параметры генерации")

        col3, col4 = st.columns(2)
        with col3:
            start_row = st.number_input("Начальная строка", min_value=0, max_value=len(df)-1, value=0)
        with col4:
            end_row = st.number_input("Конечная строка", min_value=start_row+1, max_value=len(df), value=min(start_row+10, len(df)))

        batch_size = end_row - start_row
        st.info(f"📊 Будет обработано строк: {batch_size}")

        # Кнопка генерации
        if st.button("🚀 Сгенерировать задания", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Пожалуйста, введите API ключ в боковой панели")
                return

            client = anthropic.Anthropic(api_key=api_key)

            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Обработка строк
            for idx in range(start_row, end_row):
                row = df.iloc[idx]

                # Получаем данные из строки
                competence = str(row.iloc[0])
                indicator = str(row.iloc[1])
                discipline = str(row.iloc[2])
                task_type = str(row.iloc[3])

                # Пропускаем строки с NaN в типе задания
                if pd.isna(row.iloc[3]) or task_type == 'nan':
                    status_text.warning(f"⏭️ Строка {idx+1}: пропущена (нет типа задания)")
                    continue

                # Находим соответствующий промпт
                if task_type in prompts:
                    status_text.info(f"🔄 Обработка строки {idx+1}/{end_row}...")

                    task, key = generate_task(
                        client,
                        prompts[task_type],
                        competence,
                        indicator,
                        discipline,
                        task_type
                    )

                    # Записываем результаты
                    df.at[idx, df.columns[4]] = task  # Столбец E (Задание)
                    df.at[idx, df.columns[5]] = key   # Столбец F (Ключ)

                    status_text.success(f"✅ Строка {idx+1} обработана")
                else:
                    status_text.warning(f"⚠️ Строка {idx+1}: не найден промпт для типа '{task_type}'")

                # Обновляем прогресс
                progress = (idx - start_row + 1) / batch_size
                progress_bar.progress(progress)

                # Небольшая задержка чтобы не превысить rate limit
                time.sleep(0.5)

            progress_bar.progress(1.0)
            status_text.success("🎉 Все задания сгенерированы!")

            # Показываем результаты
            st.markdown("---")
            st.subheader("📊 Результаты")
            st.dataframe(df.iloc[start_row:end_row])

            # Экспорт
            st.markdown("---")
            st.subheader("💾 Экспорт результатов")

            # Создаем Excel файл в памяти
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Задания')

            excel_data = output.getvalue()

            st.download_button(
                label="📥 Скачать результат (Excel)",
                data=excel_data,
                file_name="result_with_tasks.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    else:
        st.info("👆 Пожалуйста, загрузите оба файла для начала работы")


if __name__ == "__main__":
    main()
