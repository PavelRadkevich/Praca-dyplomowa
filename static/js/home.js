$(document).ready(function () {
    $('#companySelect').on('change', function () {
        const companySymbol = $(this).val();
        if (!companySymbol) return;

        // Clearing and disabling year selection fields before querying
        $('#startYear, #endYear').empty().append('<option value="" selected disabled>Ładowanie...</option>').prop('disabled', true);

        // Sending the request to the server
        $.ajax({
            url: `/api/get_company_years/${companySymbol}`,
            method: 'GET',
            success: function (response) {
                const years = response.years;

                // Fill in the year fields
                $('#startYear, #endYear').empty().append('<option value="" selected disabled>Wybierz rok</option>');
                years.forEach((year, index) => {
                    if (index === 0) {
                        $('#startYear, #endYear').append(`<option value="${year}" disabled>${year}</option>`);
                    } else if (index === years.length - 1) {
                        $('#startYear').append(`<option value="${year}" disabled>${year}</option>`);
                        $('#endYear').append(`<option value="${year}" selected>${year}</option>`);
                    } else {
                        $('#startYear').append(`<option value="${year}">${year}</option>`);
                        $('#endYear').append(`<option value="${year}" disabled>${year}</option>`);
                    }
                });

                // Enable date selection
                $('#startYear, #endYear').prop('disabled', false);
            },
            error: function () {
                alert('Failed to load the years data. Try again.');
                $('#startYear, #endYear').empty().append('<option value="" selected disabled>Wybierz rok</option>').prop('disabled', true);
            }
        });
    });

    $('#generateButton').on('click', function () {
        // Collecting all data about company from fields
        const selectedCompany = $('#companySelect').val();
        const startYear = $('#startYear').val();
        const endYear = $('#endYear').val();
        const selectedParametres = $('#parameters').val();

        if (selectedCompany == null || startYear == null || endYear == null || selectedParametres == null) {
            alert('Wypełnij wszystkie pola');
            return;
        }

        const requestData = {
            company: selectedCompany,
            startYear: startYear,
            endYear: endYear,
            parameters: selectedParametres
        };

        // Clearing error messages
        $('#error-message').hide().text('');

        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(requestData),
            success: function (response) {
                alert('Otrzymano odpowiedź od serwera: ' + JSON.stringify(response));
            },
            error: function (xhr) {
                let errorMessage = "";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage += xhr.responseJSON.error;
                } else {
                    errorMessage += ' Status: ' + xhr.status + ' (' + xhr.statusText + ')';
                }
                // Displaying error message on the page
                $('#error-message').text(errorMessage).show();
            }
        });
    });
});
