$(document).ready(function () {
    const socket = io.connect('http://127.0.0.1:5000');
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
                    } else if (index === 1) {
                        $('#endYear').append(`<option value="${year}" disabled>${year}</option>`);
                        $('#startYear').append(`<option value="${year}">${year}</option>`);
                    } else if (index === years.length - 1) {
                        $('#startYear').append(`<option value="${year}" disabled>${year}</option>`);
                        $('#endYear').append(`<option value="${year}">${year}</option>`);
                    } else {
                        $('#startYear, #endYear').append(`<option value="${year}">${year}</option>`);
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

        // Clearing
        $('#error-message').hide().text('');
        $('#progress').text('').css('color', 'blue').show();
        $('#last_dividend_date').text('...');
        $('#30_days').text('...');
        $('#60_days').text('...');
        $('#90_days').text('...');
        $('#loss').text('Loss: ...')
        $('#accuracy').text('Accuracy: ...')
        $('#precision').text('Precision: ...')
        $('#recall').text('Recall: ...')
        $('#auc').text('AUC: ...')
        $('#f1').text('F1: ...')
        $('#roc_auc_image').attr('src', '');
        $('#train_test_image').attr('src', '')

        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(requestData),
            success: function (response) {
                $('#progress').text('Gotowe').css('color', 'green').show();
                $('#progress').text('Gotowe').show()

                value_30 = (parseFloat(response['30_days']) * 100).toFixed(2);
                value_60 = (parseFloat(response['60_days']) * 100).toFixed(2);
                value_90 = (parseFloat(response['90_days']) * 100).toFixed(2);

                loss = (parseFloat(response['loss']) * 100).toFixed(2);
                accuracy = (parseFloat(response['accuracy']) * 100).toFixed(2);
                precision = (parseFloat(response['precision']) * 100).toFixed(2);
                recall = (parseFloat(response['recall']) * 100).toFixed(2);
                auc = (parseFloat(response['auc']) * 100).toFixed(2);
                f1 = (parseFloat(response['f1']) * 100).toFixed(2);

                $('#30_days').text(value_30 + "%");
                $('#60_days').text(value_60 + "%");
                $('#90_days').text(value_90 + "%");
                $('#roc_auc_image').attr('src', response['roc_auc_url']);
                $('#train_test_image').attr('src', response['train_test_url'])

                $('#loss').text('Loss: ' + loss + '%')
                $('#accuracy').text('Accuracy: ' + accuracy + '%')
                $('#precision').text('Precision: ' + precision + '%')
                $('#recall').text('Recall: ' + recall + '%')
                $('#auc').text('AUC: ' + auc + '%')
                $('#f1').text('F1: ' + f1 + '%')
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

    socket.on('progress', function(data) {
        $('#progress').text(data.status).show();
    });

    socket.on('last_date', function(data) {
        $('#last_dividend_date').text(data.status);
    });
});
