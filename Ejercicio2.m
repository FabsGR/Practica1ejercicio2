% Parámetros de entrenamiento
criterio_finalizacion = 'error'; % 'error' o 'epocas'
max_epocas = 1000;
tasa_aprendizaje = 0.1;
porcentaje_entrenamiento = 0.8;
num_particiones = 10; % Número de particiones

% Cargar los datos'
data_spheres = csvread('spheres1d10.csv');
entradas_spheres = data_spheres(:, 1:end-1);
salidas_spheres = data_spheres(:, end);
salir = 0;
while salir ~= 1
    % Seleccione el método de partición
    fprintf('Seleccione el método de partición:\n');
    fprintf('1. División Aleatoria 80/20\n');
    fprintf('2. División Stratified Aleatoria 80/20\n');
    fprintf('3. División en Bloques 80/20\n');
    fprintf('4. División en Bloques Estratificada 80/20\n');
    fprintf('5. División Primeros 80 para Entrenamiento y Últimos 20 \n');
    fprintf('6. 10 particiones\n');
    fprintf('7. Terminar programa........\n');
    
    metodo_particion = input('Ingrese el número correspondiente al método deseado: ');

    switch metodo_particion
        case 1
            % División Aleatoria 80/20
            num_muestras = size(entradas_spheres, 1);
            num_entrenamiento = round(porcentaje_entrenamiento * num_muestras);
            indices_entrenamiento = randperm(num_muestras, num_entrenamiento);
            indices_generalizacion = setdiff(1:num_muestras, indices_entrenamiento);
            
            % Inicialización de pesos y umbral para partición Aleatoria
            num_entradas = size(entradas_spheres, 2);
            pesos_aleatorio = rand(1, num_entradas);
            umbral_aleatorio = rand();
    
            % Entrenamiento del perceptrón para partición Aleatoria
            epoca = 1;
            error_global = inf;
    
            while (epoca <= max_epocas && error_global > 0)
                error_global = 0;
    
                for i = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para partición Aleatoria
                    indice = indices_entrenamiento(i);
                    salida = sum(entradas_spheres(indice, :) .* pesos_aleatorio) + umbral_aleatorio;
                    salida_binaria = sign(salida);
    
                    % Cálculo del error
                    error = salidas_spheres(indice) - salida_binaria;
    
                    % Actualización de pesos y umbral
                    pesos_aleatorio = pesos_aleatorio + tasa_aprendizaje * error * entradas_spheres(indice, :);
                    umbral_aleatorio = umbral_aleatorio + tasa_aprendizaje * error;
    
                    error_global = error_global + abs(error);
                end
    
                epoca = epoca + 1;
            end
    
            % Calcular la precisión en la partición Aleatoria
            precision_aleatorio = sum(salidas_spheres(indices_generalizacion) == sign(entradas_spheres(indices_generalizacion, :) * pesos_aleatorio' + umbral_aleatorio)) / length(indices_generalizacion);
    
            % Mostrar gráficamente los patrones y la recta separadora para la partición Aleatoria
            figure;
            scatter(entradas_spheres(salidas_spheres == 1, 1), ...
                    entradas_spheres(salidas_spheres == 1, 2), 'O', 'b');
            hold on;
            scatter(entradas_spheres(salidas_spheres == -1, 1), ...
                    entradas_spheres(salidas_spheres == -1, 2), 'x', 'r');
    
            x_vals = linspace(min(entradas_spheres(:, 1)), max(entradas_spheres(:, 1)));
            y_vals = -(pesos_aleatorio(1) * x_vals + umbral_aleatorio) / pesos_aleatorio(2);
            plot(x_vals, y_vals, 'g', 'LineWidth', 2);
    
            legend('Salida 1', 'Salida -1', 'Recta separadora');
            xlabel('Entrada 1');
            ylabel('Entrada 2');
            title('Patrones y Recta Separadora - Partición Aleatoria');
    
            hold off;
            
        case 2
            % División Stratified Aleatoria 80/20
            num_clases = length(unique(salidas_spheres));
            indices_entrenamiento = [];
            indices_generalizacion = [];
    
            for clase = 1:num_clases
                indices_clase = find(salidas_spheres == clase);
                num_muestras_clase = length(indices_clase);
                num_entrenamiento_clase = round(porcentaje_entrenamiento * num_muestras_clase);
                
                % Realizar una división aleatoria estratificada para cada clase
                indices_entrenamiento_clase = randsample(indices_clase, num_entrenamiento_clase);
                indices_generalizacion_clase = setdiff(indices_clase, indices_entrenamiento_clase);
                
                indices_entrenamiento = [indices_entrenamiento; indices_entrenamiento_clase];
                indices_generalizacion = [indices_generalizacion; indices_generalizacion_clase];
            end
            
            % Inicialización de pesos y umbral para partición Stratified Aleatoria
            num_entradas = size(entradas_spheres, 2);
            pesos_stratified = rand(1, num_entradas);
            umbral_stratified = rand();
    
            % Entrenamiento del perceptrón para partición Stratified Aleatoria
            epoca = 1;
            error_global = inf;
    
            while (epoca <= max_epocas && error_global > 0)
                error_global = 0;
    
                for i = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para partición Stratified Aleatoria
                    indice = indices_entrenamiento(i);
                    salida = sum(entradas_spheres(indice, :) .* pesos_stratified) + umbral_stratified;
                    salida_binaria = sign(salida);
    
                    % Cálculo del error
                    error = salidas_spheres(indice) - salida_binaria;
    
                    % Actualización de pesos y umbral
                    pesos_stratified = pesos_stratified + tasa_aprendizaje * error * entradas_spheres(indice, :);
                    umbral_stratified = umbral_stratified + tasa_aprendizaje * error;
    
                    error_global = error_global + abs(error);
                end
    
                epoca = epoca + 1;
            end
    
            % Calcular la precisión en la partición Stratified Aleatoria
            precision_stratified = sum(salidas_spheres(indices_generalizacion) == sign(entradas_spheres(indices_generalizacion, :) * pesos_stratified' + umbral_stratified)) / length(indices_generalizacion);
    
            % Mostrar gráficamente los patrones y la recta separadora para la partición Stratified Aleatoria
            figure;
            scatter(entradas_spheres(salidas_spheres == 1, 1), ...
                    entradas_spheres(salidas_spheres == 1, 2), 'O', 'b');
            hold on;
            scatter(entradas_spheres(salidas_spheres == -1, 1), ...
                    entradas_spheres(salidas_spheres == -1, 2), 'x', 'r');
    
            x_vals = linspace(min(entradas_spheres(:, 1)), max(entradas_spheres(:, 1)));
            y_vals = -(pesos_stratified(1) * x_vals + umbral_stratified) / pesos_stratified(2);
            plot(x_vals, y_vals, 'g', 'LineWidth', 2);
    
            legend('Salida 1', 'Salida -1', 'Recta separadora');
            xlabel('Entrada 1');
            ylabel('Entrada 2');
            title('Patrones y Recta Separadora - Partición Stratified Aleatoria');
    
            hold off;
            
        case 3
            % División en Bloques 80/20
            num_bloques = 5; % Dividir en 5 bloques y tomar 4 para entrenamiento, 1 para generalización
            indices_bloques = ceil((1:num_muestras) / (num_muestras / num_bloques));
            bloque_generalizacion = num_bloques; % El último bloque se usa para generalización
            indices_entrenamiento = find(indices_bloques ~= bloque_generalizacion);
            indices_generalizacion = find(indices_bloques == bloque_generalizacion);
            
            % Inicialización de pesos y umbral para partición en Bloques
            num_entradas = size(entradas_spheres, 2);
            pesos_bloques = rand(1, num_entradas);
            umbral_bloques = rand();
    
            % Entrenamiento del perceptrón para partición en Bloques
            epoca = 1;
            error_global = inf;
    
            while (epoca <= max_epocas && error_global > 0)
                error_global = 0;
    
                for i = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para partición en Bloques
                    indice = indices_entrenamiento(i);
                    salida = sum(entradas_spheres(indice, :) .* pesos_bloques) + umbral_bloques;
                    salida_binaria = sign(salida);
    
                    % Cálculo del error
                    error = salidas_spheres(indice) - salida_binaria;
    
                    % Actualización de pesos y umbral
                    pesos_bloques = pesos_bloques + tasa_aprendizaje * error * entradas_spheres(indice, :);
                    umbral_bloques = umbral_bloques + tasa_aprendizaje * error;
    
                    error_global = error_global + abs(error);
                end
    
                epoca = epoca + 1;
            end
    
            % Calcular la precisión en la partición en Bloques
            precision_bloques = sum(salidas_spheres(indices_generalizacion) == sign(entradas_spheres(indices_generalizacion, :) * pesos_bloques' + umbral_bloques)) / length(indices_generalizacion);
    
            % Mostrar gráficamente los patrones y la recta separadora para la partición en Bloques
            figure;
            scatter(entradas_spheres(salidas_spheres == 1, 1), ...
                    entradas_spheres(salidas_spheres == 1, 2), 'O', 'b');
            hold on;
            scatter(entradas_spheres(salidas_spheres == -1, 1), ...
                    entradas_spheres(salidas_spheres == -1, 2), 'x', 'r');
    
            x_vals = linspace(min(entradas_spheres(:, 1)), max(entradas_spheres(:, 1)));
            y_vals = -(pesos_bloques(1) * x_vals + umbral_bloques) / pesos_bloques(2);
            plot(x_vals, y_vals, 'g', 'LineWidth', 2);
    
            legend('Salida 1', 'Salida -1', 'Recta separadora');
            xlabel('Entrada 1');
            ylabel('Entrada 2');
            title('Patrones y Recta Separadora - Partición en Bloques');
    
            hold off;
            
        case 4
            % División en Bloques Estratificada 80/20
            num_bloques = 5; % Dividir en 5 bloques y tomar 4 para entrenamiento, 1 para generalización
            num_clases = length(unique(salidas_spheres));
            indices_entrenamiento = [];
            indices_generalizacion = [];
    
            for clase = 1:num_clases
                indices_clase = find(salidas_spheres == clase);
                num_muestras_clase = length(indices_clase);
                
                % División 80/20 en bloques para cada clase
                indices_bloques = ceil((1:num_muestras_clase) / (num_muestras_clase / num_bloques));
                bloque_generalizacion = num_bloques; % El último bloque se usa para generalización
                indices_entrenamiento_clase = find(indices_bloques ~= bloque_generalizacion);
                indices_generalizacion_clase = find(indices_bloques == bloque_generalizacion);
                
                % Mapear los índices de bloques de clase a índices de muestras globales
                indices_entrenamiento = [indices_entrenamiento; indices_clase(indices_entrenamiento_clase)];
                indices_generalizacion = [indices_generalizacion; indices_clase(indices_generalizacion_clase)];
            end
            
            % Inicialización de pesos y umbral para partición en Bloques Estratificada
            num_entradas = size(entradas_spheres, 2);
            pesos_bloques_estratificada = rand(1, num_entradas);
            umbral_bloques_estratificada = rand();
    
            % Entrenamiento del perceptrón para partición en Bloques Estratificada
            epoca = 1;
            error_global = inf;
    
            while (epoca <= max_epocas && error_global > 0)
                error_global = 0;
    
                for i = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para partición en Bloques Estratificada
                    indice = indices_entrenamiento(i);
                    salida = sum(entradas_spheres(indice, :) .* pesos_bloques_estratificada) + umbral_bloques_estratificada;
                    salida_binaria = sign(salida);
    
                    % Cálculo del error
                    error = salidas_spheres(indice) - salida_binaria;
    
                    % Actualización de pesos y umbral
                    pesos_bloques_estratificada = pesos_bloques_estratificada + tasa_aprendizaje * error * entradas_spheres(indice, :);
                    umbral_bloques_estratificada = umbral_bloques_estratificada + tasa_aprendizaje * error;
    
                    error_global = error_global + abs(error);
                end
    
                epoca = epoca + 1;
            end
    
            % Calcular la precisión en la partición en Bloques Estratificada
            precision_bloques_estratificada = sum(salidas_spheres(indices_generalizacion) == sign(entradas_spheres(indices_generalizacion, :) * pesos_bloques_estratificada' + umbral_bloques_estratificada)) / length(indices_generalizacion);
    
            % Mostrar gráficamente los patrones y la recta separadora para la partición en Bloques Estratificada
            figure;
            scatter(entradas_spheres(salidas_spheres == 1, 1), ...
                    entradas_spheres(salidas_spheres == 1, 2), 'O', 'b');
            hold on;
            scatter(entradas_spheres(salidas_spheres == -1, 1), ...
                    entradas_spheres(salidas_spheres == -1, 2), 'x', 'r');
    
            x_vals = linspace(min(entradas_spheres(:, 1)), max(entradas_spheres(:, 1)));
            y_vals = -(pesos_bloques_estratificada(1) * x_vals + umbral_bloques_estratificada) / pesos_bloques_estratificada(2);
            plot(x_vals, y_vals, 'g', 'LineWidth', 2);
    
            legend('Salida 1', 'Salida -1', 'Recta separadora');
            xlabel('Entrada 1');
            ylabel('Entrada 2');
            title('Patrones y Recta Separadora - Partición en Bloques Estratificada');
    
            hold off;
    
        case 5
            % División Primeros 80% para Entrenamiento y Últimos 20% para Generalización
            porcentaje_entrenamiento = 0.8;
            num_muestras = size(entradas_spheres, 1);
            num_entrenamiento = round(porcentaje_entrenamiento * num_muestras);
            indices_entrenamiento = 1:num_entrenamiento;
            indices_generalizacion = (num_entrenamiento + 1):num_muestras;
        
            % Inicialización de pesos y umbral para esta partición
            num_entradas = size(entradas_spheres, 2);
            pesos_particion_5 = rand(1, num_entradas);
            umbral_particion_5 = rand();
        
            % Entrenamiento del perceptrón para esta partición
            epoca = 1;
            error_global = inf;
        
            while (epoca <= max_epocas && error_global > 0)
                error_global = 0;
        
                for i = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para esta partición
                    indice = indices_entrenamiento(i);
                    salida = sum(entradas_spheres(indice, :) .* pesos_particion_5) + umbral_particion_5;
                    salida_binaria = sign(salida);
        
                    % Cálculo del error
                    error = salidas_spheres(indice) - salida_binaria;
        
                    % Actualización de pesos y umbral
                    pesos_particion_5 = pesos_particion_5 + tasa_aprendizaje * error * entradas_spheres(indice, :);
                    umbral_particion_5 = umbral_particion_5 + tasa_aprendizaje * error;
        
                    error_global = error_global + abs(error);
                end
        
                epoca = epoca + 1;
            end
        
            % Calcular la precisión en esta partición
            precision_particion_5 = sum(salidas_spheres(indices_generalizacion) == sign(entradas_spheres(indices_generalizacion, :) * pesos_particion_5' + umbral_particion_5)) / length(indices_generalizacion);
        
            % Mostrar gráficamente los patrones y la recta separadora para esta partición
            figure;
            scatter(entradas_spheres(salidas_spheres == 1, 1), ...
                    entradas_spheres(salidas_spheres == 1, 2), 'O', 'b');
            hold on;
            scatter(entradas_spheres(salidas_spheres == -1, 1), ...
                    entradas_spheres(salidas_spheres == -1, 2), 'x', 'r');
        
            x_vals = linspace(min(entradas_spheres(:, 1)), max(entradas_spheres(:, 1)));
            y_vals = -(pesos_particion_5(1) * x_vals + umbral_particion_5) / pesos_particion_5(2);
            plot(x_vals, y_vals, 'g', 'LineWidth', 2);
        
            legend('Salida 1', 'Salida -1', 'Recta separadora');
            xlabel('Entrada 1');
            ylabel('Entrada 2');
            title('Partición Primeros 80% para Entrenamiento y Últimos 20%');
        
            hold off;
    
        case 6
           % Cargar los datos de los archivos CSV
            data1 = csvread('spheres2d10.csv');
            data2 = csvread('spheres2d50.csv');
            data3 = csvread('spheres2d70.csv');
            
            % Concatenar los datos en un solo conjunto de datos
            data = [data1; data2; data3];
            
            % Dividir los datos en 10 particiones 80-20 del metodo K-fold
            numParticiones = 10;
            particiones = cvpartition(size(data, 1), 'KFold', numParticiones);
            
            % Inicializar una variable para almacenar las precisiones
            precisiones = zeros(numParticiones, 1);
            
            % Inicializar pesos y umbral de manera aleatoria
            pesos_particion = rand(1, 3);  % Inicializados aleatoriamente
            umbral_particion = rand();  % Inicializado aleatoriamente
            
            epoca = 1;
            error_global = Inf;
            indices_entrenamiento = randperm(size(data, 1));

            
            % Entrenamiento del perceptrón utilizando todos los datos de entrenamiento
            while (epoca <= max_epocas && any(error_global > 0))
                error_global = 0;  % Inicializar error_global en cada época
                
                
                for j = 1:length(indices_entrenamiento)
                    % Cálculo de la salida del perceptrón para esta partición
                    indice = indices_entrenamiento(j);
                    entrada = data(indice, 1:3);
                    salida = sum(entrada .* pesos_particion) + umbral_particion;
                    salida_binaria = sign(salida);
                    
                    % Cálculo del error
                    error = data(indice, 4) - salida_binaria;
                    
                    % Actualización de pesos y umbral
                    pesos_particion = pesos_particion + tasa_aprendizaje * error * entrada';
                    umbral_particion = umbral_particion + tasa_aprendizaje * error;
                    
                    error_global = error_global + abs(error);
                end
                
                epoca = epoca + 1;
            end
            
            
            % Mostrar la gráfica de los datos y la línea divisoria en 2D
            X_projected = data(:, 1:3);
            
         % Calcular la línea divisoria en 3D
            [x, y] = meshgrid(min(X_projected(:, 1)):0.01:max(X_projected(:, 1)), min(X_projected(:, 2)):0.01:max(X_projected(:, 2)));
            z = (-pesos_particion(1) .* x - pesos_particion(2) .* y - umbral_particion(2)) / pesos_particion(3);
            
            % Mostrar la gráfica de los datos y la línea divisoria en 3D
            figure;
            
            % Graficar los datos en 3D
            scatter3(X_projected(:, 1), X_projected(:, 2), data(:, 3), 20, data(:, 4), 'filled');
            hold on;
            
            % Graficar la línea divisoria en 3D
            surf(x, y, z, 'EdgeColor', 'none');
            
            xlabel('Componente Principal 1');
            ylabel('Componente Principal 2');
            zlabel('Característica 3');
            title('Línea divisoria del perceptrón en 3D');
            
            % Mostrar la leyenda
            legend('Datos', 'Línea divisoria');


        case 7
            disp("Termino el programa");
            salir = 1;
    
            
        otherwise
            disp('Opción no válida. Seleccione un método de partición válido.');
            return;
    end
end