$(document).ready(function () {
    $('#companySelect').on('change', function () {
        const companySymbol = $(this).val();
        if (!companySymbol) return;

        // Очистка и выключение полей выбора года перед запросом
        $('#startYear, #endYear').empty().append('<option value="" selected disabled>Загрузка...</option>').prop('disabled', true);

        // Отправляем запрос на сервер
        $.ajax({
            url: `/api/get_company_years/${companySymbol}`,
            method: 'GET',
            success: function (response) {
                const years = response.years;

                // Заполняем поля годов
                $('#startYear, #endYear').empty().append('<option value="" selected disabled>Выберите год</option>');
                years.forEach(year => {
                    $('#startYear, #endYear').append(`<option value="${year}">${year}</option>`);
                });

                // Включаем выбор дат
                $('#startYear, #endYear').prop('disabled', false);
            },
            error: function () {
                alert('Failed to load the years data. Try again.');
                $('#startYear, #endYear').empty().append('<option value="" selected disabled>Wybierz rok</option>').prop('disabled', true);
            }
        });
    });
});
